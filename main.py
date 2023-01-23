import contextlib
import datetime
import enum
import functools
import logging
import os
import pathlib
import pickle
import sys
import time
import typing

import cytomine
import numpy as np
import shapely.wkt
from cytomine import models
from shapely.affinity import affine_transform
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


RetType = typing.TypeVar("RetType")


def retry(
    fun: typing.Callable[[], typing.Optional[RetType]],
    delay: float = 1.0,
    max_retry: int = 5,
    call_back: typing.Callable[[int], None] = None,
):
    "retry if the result of fun is None"

    for retry_ in range(max_retry):
        time.sleep(0.0 if retry_ == 0 else delay)

        val = fun()
        if val is not None:
            return val

        if call_back is not None:
            call_back(retry_)

    return None


@contextlib.contextmanager
def no_output():
    try:
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stderr(devnull):
                with contextlib.redirect_stdout(devnull):
                    yield None
    finally:
        pass


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
    registration_type: RegistrationType
    compose_non_rigid: bool

    # pixel dims for registration (defines how scaled down they are)
    micro_reg_max_dim_px: typing.Optional[int]

    annotations_to_map: typing.List[models.Annotation]
    images_to_warp: typing.List[models.ImageInstance]
    upload_host: typing.Optional[str]
    map_annotations_to_warped_images: bool

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

        registration_type = RegistrationType.NON_RIGID
        if has("registration_type"):
            registration_type = RegistrationType(namespace.registration_type)

        compose_non_rigid = False
        if has("compose_non_rigid"):
            compose_non_rigid = eb(namespace.compose_non_rigid)

        micro_reg_max_dim_px = None
        if has("micro_reg_max_dim_px"):
            micro_reg_max_dim_px = ei(namespace.micro_reg_max_dim_px)

        if (
            micro_reg_max_dim_px is not None
            and registration_type != RegistrationType.MICRO
        ):
            raise ValueError(
                "can only specify MICRO_REG_MAX_DIM if " "REGISTRATION_TYPE is 'micro'"
            )

        annotation_to_map_ids = []
        if has("annotation_to_map"):
            annotation_to_map_ids = [
                ei(i) for i in namespace.annotation_to_map.split(",")
            ]

        images_to_warp_ids = []
        if has("images_to_warp"):
            images_to_warp_ids = [ei(i) for i in namespace.images_to_warp.split(",")]

        upload_host = None
        if has("cytomine_upload_host"):
            upload_host = namespace.cytomine_upload_host

        if images_to_warp_ids and not upload_host:
            raise ValueError(
                "must submit 'cytomine_upload_host' if there are "
                "images to deform and upload"
            )

        map_annotations_to_warped_images = False
        if has("map_annotations_to_warped_images"):
            map_annotations_to_warped_images = eb(
                namespace.map_annotations_to_warped_images
            )

        if ref_image_id is not None and ref_image_id not in all_image_ids:
            all_image_ids.append(ref_image_id)

        if any(img not in all_image_ids for img in images_to_warp_ids):
            raise ValueError(
                "all images to warp should be part of the registration sequence"
            )

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
            registration_type=registration_type,
            compose_non_rigid=compose_non_rigid,
            micro_reg_max_dim_px=micro_reg_max_dim_px,
            annotations_to_map=[ann_cache[id] for id in annotation_to_map_ids],
            images_to_warp=[img_cache[idx] for idx in images_to_warp_ids],
            upload_host=upload_host,
            map_annotations_to_warped_images=map_annotations_to_warped_images,
        )

    def __repr__(self) -> str:
        asdict = self._asdict()
        if self.reference_image is not None:
            asdict["reference_image"] = self.reference_image.id
        asdict["all_images"] = [img.id for img in self.all_images]
        asdict["annotations_to_map"] = [a.id for a in self.annotations_to_map]
        asdict["images_to_warp"] = [img.id for img in self.images_to_warp]

        return pretty_repr(asdict)


def pretty_repr(o: typing.Any) -> str:
    if isinstance(o, (int, float, str, type(None))):
        return f"{o!r}"

    if isinstance(o, enum.Enum):
        return type(o).__name__ + "." + o.name

    # named tuple
    if isinstance(o, tuple) and hasattr(o, "_asdict"):
        return f"{o!r}"

    if isinstance(o, typing.Mapping):
        inner = (pretty_repr(k) + ":" + pretty_repr(v) for k, v in o.items())
        return "{" + ", ".join(inner) + "}"

    if isinstance(o, typing.Iterable):
        return "[" + ", ".join(pretty_repr(s) for s in o) + "]"

    if hasattr(o, "__dict__"):
        return str(type(o)) + ":" + pretty_repr(vars(o))

    if hasattr(o, "__slots__"):
        inner = pretty_repr({attr: getattr(o, attr) for attr in o.__slots__})
        return str(type(o)) + ":" + inner

    # default representation
    return f"{o!r}"


class VALISJob(typing.NamedTuple):

    cytomine_job: cytomine.CytomineJob
    parameters: JobParameters
    src_dir: pathlib.Path
    dst_dir: pathlib.Path
    name: str
    logger: logging.Logger = logging.getLogger("VALISJob")

    def update(self, progress: int, status: str):
        self.cytomine_job.job.update(
            status=models.Job.RUNNING, progress=progress, statusComment=status
        )

    def run(self):
        # logger config
        self.logger.setLevel(logging.INFO)
        self.logger.info(pretty_repr(self.parameters))

        self.update(1, "Parsed parameters")

        with contextlib.ExitStack() as stack:
            self.update(1, "Initializing JVM")
            registration.init_jvm()
            stack.callback(registration.kill_jvm)
            self.update(2, "Initialized JVM")

            self.update(2, "Downloading all images")
            self.download_all_images()
            self.update(20, "Downloaded images")

            self.update(20, "Creating registrar")
            # NOTE Valis doesn't allow 'lazy' folders: the images must be downloaded before creating the registrar
            registrar = registration.Valis(**self.get_valis_args())
            self.update(21, "Created registrar")

            self.update(21, "Registering all images")
            self.register(registrar)
            self.update(60, "Registered all images")

            self.update(60, "Warping all annotations")
            self.warp_annotations(registrar, to_reference=True)
            self.update(70, "Warped all annotations")

            self.update(70, "Warping images")
            img_lst = self.warp_images(registrar)
            self.update(89, "Warped all images")

            if self.parameters.map_annotations_to_warped_images:
                self.update(89, "Warping annotations to uploaded images")
                self.warp_annotations(registrar, img_lst)
                self.update(99, "Warped all annotations to uploaded images")

    def get_valis_args(self):
        valis_args = {
            "src_dir": str(self.src_dir),
            "dst_dir": str(self.dst_dir),
            "name": self.name,
            "imgs_ordered": self.parameters.image_ordering != ImageOrdering.AUTO,
            "compose_non_rigid": self.parameters.compose_non_rigid,
            "align_to_reference": not self.parameters.align_toward_reference,
            "crop": self.parameters.image_crop.value,
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
        fname = "{app}-{utc}_{name}".format(
            app=self.cytomine_job.software.id,
            utc=datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S"),
            name=image.filename,
        )
        return str((self.dst_dir / "saved-images" / fname).with_suffix(".ome.tiff"))

    def download_all_images(self):
        # TODO can this be parallelized ?

        for image in self.parameters.all_images:
            dest_path = self.get_image_path(image)
            if pathlib.Path(dest_path).exists():
                continue
            status = image.download(dest_path, override=False)
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
        with no_output():
            rigid_registrar, non_rigid_registrar, _ = registrar.register()
        micro_registrar = None

        assert rigid_registrar is not None

        self.logger.info("non-micro registration done")

        # attach registrar to Job in Cytomine (automatically pickled by VALIS)
        # registrar_path = self.registrar_path()
        # models.AttachedFile(
        #     self.cytomine_job.job,
        #     domainIndent=self.cytomine_job.job.id,
        #     filename=registrar_path,
        #     domainClassName="be.cytomine.processing.Job",
        # ).upload()

        if self.parameters.registration_type == RegistrationType.MICRO:
            with no_output():
                if self.parameters.micro_reg_max_dim_px is not None:
                    micro_registrar, _ = registrar.register_micro(
                        max_non_rigid_registartion_dim_px=self.parameters.micro_reg_max_dim_px
                    )
                else:
                    micro_registrar, _ = registrar.register_micro()
            # micro_registrar_path = self.registrar_path("micro")

            self.logger.info("micro registration done")

            # # pickle it
            # with open(micro_registrar_path, "wb") as micro_registrar_dest:
            #     pickle.dump(micro_registrar, micro_registrar_dest)

            # models.AttachedFile(
            #     self.cytomine_job.job,
            #     domainIndent=self.cytomine_job.job.id,
            #     filename=micro_registrar_path,
            #     domainClassName="be.cytomine.processing.Job",
            # ).upload()

        return rigid_registrar, non_rigid_registrar, micro_registrar

    def warp_annotations(
        self,
        registrar: registration.Valis,
        to_images: typing.Optional[typing.Iterable[models.ImageInstance]] = None,
        to_reference: bool = False,
    ):
        if to_reference and to_images:
            raise ValueError(
                "cannot warp the annotation to the reference and \
                other images in a single call to warp_annotations"
            )
        if not to_reference and not to_images:
            raise ValueError("either to_reference is True, or dest_imgs is non empty")

        if not self.parameters.annotations_to_map:
            return

        # create an output collection
        # warp all annotation that aren't on the reference image

        reference_image = self.get_reference_image(registrar)
        reference_slide = registrar.get_slide(self.get_image_path(reference_image))

        annotations = models.AnnotationCollection()

        image_cache = {int(img.id): img for img in self.parameters.all_images}

        if to_reference:
            to_images = [reference_image]
        else:
            for idx, img in enumerate(to_images):
                if not isinstance(img, (models.ImageInstance)):
                    raise ValueError(f"img at {idx=} is not an ImageInstance")

            if any(img.id in image_cache for img in to_images):
                raise ValueError(
                    "when warping to non-reference image, only \
                    new images are supported"
                )

        def warper(src: registration.Slide, x, y, z=None):
            if z is not None:
                raise ValueError("unable to warp 3D points")

            xy = np.stack([x, y], axis=1)
            if to_reference:
                warped_xy = src.warp_xy_from_to(xy, reference_slide)
            else:
                warped_xy = src.warp_xy(xy, crop=True)

            return warped_xy[:, 0], warped_xy[:, 1]

        for annotation in self.parameters.annotations_to_map:
            image = image_cache[annotation.image]
            slide: registration.Slide = registrar.get_slide(self.get_image_path(image))

            # get annotation geometry
            geometry = shapely.wkt.loads(annotation.location)

            # convert to top-left coordinate
            geometry = affine_transform(geometry, [1, 0, 0, -1, 0, image.height])

            # warp points
            geometry = transform(functools.partial(warper, slide), geometry)

            for img in to_images:
                if img.id == annotation.image:
                    continue  # avoid duplicate annotations

                # convert back to bottom-left coordinate
                geometry_local = affine_transform(
                    geometry, [1, 0, 0, -1, 0, img.height]
                )

                geometry_str = shapely.wkt.dumps(geometry_local)

                annotations.append(
                    models.Annotation(
                        geometry_str,
                        img.id,
                        annotation.term,
                        annotation.project,
                    )
                )

        self.logger.info("pushing %d annotations", len(annotations))
        annotations.save()

    def warp_images(self, registrar: registration.Valis):

        if not self.parameters.images_to_warp:
            uploaded_images: typing.List[models.ImageInstance] = []
            return uploaded_images

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
            with no_output():
                # remove progress bar
                slide.warp_and_save_slide(path_dst, tile_wh=1024)
            self.logger.info("warped %s", path_dst)

            warped_images.append(path_dst)

        uploaded_files: typing.List[models.UploadedFile] = []

        for path_dst in warped_images:
            uf = self.cytomine_job.upload_image(
                upload_host=self.parameters.upload_host,
                filename=path_dst,
                id_storage=user_storage.id,
                id_project=self.cytomine_job.project.id,
                sync=True,
            )
            if not uf:
                self.logger.error("failed to upload image %s", path_dst)
            else:
                uploaded_files.append(uf)

        return self.get_image_instances(uploaded_files)

    def get_image_instances(self, uploaded_files: typing.List[models.UploadedFile]):
        # IMPORTANT: API doesn't do instant modification, retry with time delay is necessary

        base_ids: typing.List[int] = []
        for uf in uploaded_files:

            def _get_abstract_image():
                response_ = self.cytomine_job.get(
                    f"{uf.callback_identifier}/{uf.id}/abstractimage.json"
                )
                if not response_:
                    return None
                return response_

            response = retry(_get_abstract_image, delay=1.5, max_retry=20)
            if response:
                base_ids.append(response["id"])
            else:
                self.logger.error("giving up on UF:%d", uf.id)

        def _get_lst():
            images = models.ImageInstanceCollection().fetch_with_filter(
                "project", self.cytomine_job.project.id
            )
            images_by_base: typing.Mapping[int, models.ImageInstance] = {
                image.baseImage: image for image in images
            }

            try:
                return [images_by_base[idx] for idx in base_ids]
            except KeyError as e:
                self.logger.exception(e)
                return None

        lst = retry(_get_lst)
        if lst is None:
            raise ValueError("impossible to fetch the list of images")
        
        if len(lst) == 0:
            raise ValueError("fetched empty list of images")

        return lst


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
