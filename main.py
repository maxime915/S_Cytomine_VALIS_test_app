import contextlib
import enum
import functools
import logging
import pathlib
import pickle
import sys
import typing

import cytomine
import numpy as np
import shapely.wkt
from cytomine import models
from shapely.ops import transform
from valis import registration


class ImageOrdering(enum.Enum):
    AUTO = "auto"
    NAME = "name"
    CREATED = "created"


class ImageCrop(enum.Enum):
    REFERENCE = "reference"
    ALL = "all"
    OVERLAP = "overlap"


class RegistrationType(enum.Enum):
    RIGID = "rigid"
    NON_RIGID = "non-rigid"
    MICRO = "micro"


def ei(val: typing.Union[str, int]) -> int:
    "expect int"
    if isinstance(val, int):
        return int(val)  # make sure bool values are converted to int
    if isinstance(val, str) and str(int(val)) == val:
        return int(val)
    raise ValueError(f"{val=!r} is not an int")


def eb(val: typing.Union[str, bool]) -> bool:
    "expect bool"
    if isinstance(val, bool):
        return val
    if isinstance(val, str) and str(bool(val)).lower() == val.lower():
        return bool(val)
    raise ValueError(f"{val=!r} is not a bool")


class JobParameters(typing.NamedTuple):
    """"""
    
    # IDEA: previous Job ID to pick the registrar used ?

    # IDEA: img_type: ["brightfield", "fluorescence", "multi", None]
    # img_type appears to not do anything when looking at VALIS code

    reference_image: typing.Optional[models.ImageInstance]
    all_images: typing.List[models.ImageInstance]
    image_ordering: ImageOrdering
    align_toward_reference: bool
    image_crop: ImageCrop
    # check_for_reflections: bool  # FIXME depends on the version of Valis
    registration_type: RegistrationType
    compose_non_rigid: bool

    # IDEA: pixel dims for registration (defines how scaled down they are)

    # TODO: either select all userAnnotation in a project
    #   or select all annotations from a list of users (and filter by images)
    annotations_to_map: typing.List[models.Annotation]
    images_to_warp: typing.List[models.ImageInstance]
    # FIXME not ready
    # map_annotations_to_warped_images: bool

    @staticmethod
    def check(namespace: typing.Mapping[str, typing.Any], project: models.Project):
        "raise ValueError on bad parameters"

        def has(key: str):
            return key in namespace and getattr(namespace, key) is not None

        ref_image_id: typing.Optional[int] = None
        if has("reference_image"):
            ref_image_id = ei(namespace.reference_image)

        all_image_ids = [ei(i) for i in namespace.all_images.split(",")]

        image_ordering = ImageOrdering.AUTO
        if has("image_ordering"):
            image_ordering = ImageOrdering(namespace.image_ordering)

        align_toward_reference = True
        if has("align_toward_reference"):
            align_toward_reference = eb(namespace.align_toward_reference)

        image_crop = ImageCrop(namespace.image_crop)

        # FIXME Valis version mismatch after env solve
        # check_for_reflections = False
        # if has("check_for_reflections"):
        #     check_for_reflections = eb(namespace.check_for_reflections)

        registration_type = RegistrationType.NON_RIGID
        if has("registration_type"):
            registration_type = RegistrationType(namespace.registration_type)

        compose_non_rigid = False
        if has("compose_non_rigid"):
            compose_non_rigid = eb(namespace.compose_non_rigid)

        annotation_to_map_ids = []
        if has("annotation_to_map"):
            annotation_to_map_ids = [ei(i) for i in namespace.annotation_to_map.split(",")]

        images_to_warp_ids = []
        if has("images_to_warp"):
            images_to_warp_ids = [ei(i) for i in namespace.images_to_warp.split(",")]

        # map_annotations_to_warped_images = False
        # if has("map_annotations_to_warped_images"):
        #     map_annotations_to_warped_images = eb(namespace.map_annotations_to_warped_images)

        if ref_image_id is not None and ref_image_id not in all_image_ids:
            all_image_ids.append(ref_image_id)

        if any(img not in all_image_ids for img in images_to_warp_ids):
            raise ValueError(
                "all images to warp should be part of the registration sequence"
            )

        # TODO how to fetch a list of ids in O(1) requests ?
        all_images_in_project = models.ImageInstanceCollection().fetch_with_filter(
            "project", project.id
        )
        all_annotations_in_project = models.AnnotationCollection()
        all_annotations_in_project.project = project.id
        all_annotations_in_project.showWKT = True
        all_annotations_in_project.showTerm = True
        all_annotations_in_project.fetch()

        img_cache = {img.id: img for img in all_images_in_project}
        ann_cache = {ann.id: ann for ann in all_annotations_in_project}

        if any(ann.image not in all_image_ids for ann in ann_cache.values()):
            raise ValueError(
                "annotations may only come from images in the registration sequence"
            )

        return JobParameters(
            reference_image=img_cache.get(ref_image_id, None),
            all_images=[img_cache[idx] for idx in all_image_ids],
            image_ordering=image_ordering,
            align_toward_reference=align_toward_reference,
            image_crop=image_crop,
            # check_for_reflections=check_for_reflections,
            registration_type=registration_type,
            compose_non_rigid=compose_non_rigid,
            annotations_to_map=[ann_cache[id] for id in annotation_to_map_ids],
            images_to_warp=[img_cache[idx] for idx in images_to_warp_ids],
            # map_annotations_to_warped_images=map_annotations_to_warped_images,
        )


class VALISJob(typing.NamedTuple):

    cytomine_job: cytomine.CytomineJob
    parameters: JobParameters
    src_dir: pathlib.Path
    dst_dir: pathlib.Path
    name: str

    def update(self, progress: int, status: str):
        self.cytomine_job.job.update(
            status=models.Job.RUNNING, progress=progress, status_comment=status
        )

    def run(self):
        self.update(1, "Parsed parameters")

        with contextlib.ExitStack() as stack:
            registration.init_jvm()
            stack.callback(lambda: registration.kill_jvm())
            self.update(2, "Initialized JVM")

            self.download_all_images()
            self.update(20, "Downloaded images")

            # NOTE Valis doesn't allow 'lazy' folders: the images must be downloaded before creating the registrar
            registrar = registration.Valis(**self.get_valis_args())
            self.update(21, "Created registrar")

            self.register(registrar)
            self.update(60, "Registered all images")

            self.warp_annotations(registrar)
            self.update(70, "Warped all annotations")

            self.warp_images(registrar)
            self.update(99, "Warped all images")

    def get_valis_args(self):
        valis_args = {
            "src_dir": str(self.src_dir),
            "dst_dir": str(self.dst_dir),
            "name": self.name,
            "imgs_ordered": self.parameters.image_ordering != ImageOrdering.AUTO,
            "compose_non_rigid": self.parameters.compose_non_rigid,
            "align_to_reference": not self.parameters.align_toward_reference,
            "crop": self.parameters.image_crop.value,
            # "check_for_reflections": self.parameters.check_for_reflections,
        }

        # skip non rigid registrations
        if self.parameters.registration_type == RegistrationType.RIGID:
            valis_args["non_rigid_registrar_cls"] = None

        # reference image
        if self.parameters.reference_image is not None:
            valis_args["reference_img_f"] = self.get_image_path(
                self.parameters.reference_image
            )

        return valis_args

    def get_image_path(self, image: models.ImageInstance) -> str:
        fname = image.filename
        if self.parameters.image_ordering == ImageOrdering.CREATED:
            fname = f"{image.created}_{image.filename}"
        return str(self.src_dir / fname)

    def get_warped_image_path_ome_tiff(self, image: models.ImageInstance) -> str:
        fname = f"{self.cytomine_job.software.id}_" + image.filename
        return str((self.dst_dir / "saved-images" / fname).with_suffix(".ome.tiff"))

    def download_all_images(self):
        # TODO can this be parallelized ?

        for image in self.parameters.all_images:
            status = image.download(self.get_image_path(image), override=False)
            if not status:
                raise ValueError(f"image with ID {image.id} could not be downloaded")

    def get_reference_image(
        self, registrar: registration.Valis
    ) -> models.ImageInstance:

        if self.parameters.reference_image is not None:
            return self.parameters.reference_image

        if registrar.reference_img_f is None:
            raise ValueError(
                "registrar has no reference image yet, perform registration first"
            )

        for image in self.parameters.all_images:
            if self.get_image_path(image) == registrar.reference_img_f:
                return image

        assert False, "The image should have been found"

    def registrar_path(self, suffix: str = "") -> str:
        base = self.dst_dir / self.name / "data"
        if not suffix:
            return str(base / f"{self.name}_registrar.pickle")
        return str(base / f"{self.name}_{suffix}_registrar.pickle")

    def register(self, registrar: registration.Valis):

        # rigid and non-rigid registration
        rigid_registrar, non_rigid_registrar, _ = registrar.register()
        micro_registrar = None
        
        assert rigid_registrar is not None

        # attach registrar to Job in Cytomine (automatically pickled by VALIS)
        registrar_path = self.registrar_path()
        models.AttachedFile(
            self.cytomine_job.job,
            domainIndent=self.cytomine_job.job.id,
            filename=registrar_path,
            domainClassName="be.cytomine.processing.Job",
        ).upload()

        if self.parameters.registration_type == RegistrationType.MICRO:
            micro_registrar, _ = registrar.register_micro()
            micro_registrar_path = self.registrar_path("micro")

            # pickle it
            with open(micro_registrar_path, "wb") as micro_registrar_dest:
                pickle.dump(micro_registrar, micro_registrar_dest)

            models.AttachedFile(
                self.cytomine_job.job,
                domainIndent=self.cytomine_job.job.id,
                filename=micro_registrar_path,
                domainClassName="be.cytomine.processing.Job",
            ).upload()

        return rigid_registrar, non_rigid_registrar, micro_registrar

    def warp_annotations(self, registrar: registration.Valis):
        logging.warning("skipped")
        return
        if not self.parameters.annotations_to_map:
            return

        # create an output collection
        # warp all annotation that aren't on the reference image to the reference image

        reference_image = self.get_reference_image(registrar)
        reference_slide = registrar.get_slide(self.get_image_path(reference_image))

        annotations = models.AnnotationCollection()
        
        image_cache = {int(img.id): img for img in self.parameters.all_images}

        for annotation in self.parameters.annotations_to_map:
            image = image_cache[annotation.image]
            slide: registration.Slide = registrar.get_slide(self.get_image_path(image))

            # get annotation geometry
            geometry = shapely.wkt.loads(annotation.location)

            # warp annotation
            def _warper(x, y, z=None):
                if z is not None:
                    raise ValueError("unable to warp 3D points")

                warped_xy = slide.warp_xy_from_to(
                    np.stack([x, y], axis=1), reference_slide
                )
                return warped_xy[:, 0], warped_xy[:, 1]

            warped_geometry = transform(_warper, geometry)
            annotations.append(
                models.Annotation(
                    shapely.wkt.dumps(warped_geometry),
                    reference_image.id,
                    annotation.term,
                    annotation.project,
                )
            )

        logging.info("pushing %d annotations", len(annotations))
        annotations.save()

    def warp_images(self, registrar: registration.Valis):
        logging.warning("skipped")
        return
        if not self.parameters.images_to_warp:
            return
        
        # TODO add upload_host as a parameter

        # get storage
        userJob = models.UserJob().fetch(id=self.cytomine_job.job.userJob)
        all_storage = models.StorageCollection().fetch()
        user_storage = [s for s in all_storage if s.user == userJob.user][0]
        
        # warp all images and save to disk
        warped_images: typing.List[str] = []
        for image in self.parameters.images_to_warp:
            path_src = self.get_image_path(image)
            path_dst = self.get_warped_image_path_ome_tiff(image)

            slide: registration.Slide = registrar.get_slide(path_src)
            slide.warp_and_save_slide(path_dst)
            
            warped_images.append(path_dst)
        
        for path_dst in warped_images:
            uf = self.cytomine_job.upload_image(
                upload_host="https://research-upload.cytomine.be/",
                filename=path_dst,
                id_storage=user_storage.id,
                id_project=self.cytomine_job.project.id,
            )




def main(arguments):
    with cytomine.CytomineJob.from_cli(arguments) as job:

        job.job.update(
            status=models.Job.RUNNING, progress=0, status_comment="Initialization"
        )

        src_dir = pathlib.Path(f"./valis-slides-{job.software.id}")
        dst_base_dir = pathlib.Path(f"./valis-results-{job.software.id}")

        dst_dir = dst_base_dir / "internal"
        name = "main"

        # check all parameters and fetch from Cytomine
        parameters = JobParameters.check(job.parameters, job._project)

        src_dir.mkdir(exist_ok=True, parents=False)
        dst_base_dir.mkdir(exist_ok=True, parents=False)

        VALISJob(job, parameters, src_dir, dst_dir, name).run()

        job.job.update(
            status=models.Job.TERMINATED, progress=100, status_comment="Job terminated"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
