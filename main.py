import contextlib
import datetime
import enum
import logging
import os
import pathlib
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

def eil(val: str) -> typing.List[int]:
    if not val:
        return []
    return [ei(p) for p in val.split(",")]


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
    call_back: typing.Optional[typing.Callable[[int], None]] = None,
) -> typing.Optional[RetType]:
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
    max_proc_size: int
    micro_max_proc_size: int

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

        all_image_ids = eil(namespace.all_images)

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

        max_proc_size = registration.DEFAULT_MAX_PROCESSED_IMG_SIZE
        if has("max_proc_size"):
            max_proc_size = ei(namespace.max_proc_size)

        micro_max_proc_size = registration.DEFAULT_MAX_NON_RIGID_REG_SIZE
        if registration_type != RegistrationType.MICRO:
            micro_max_proc_size = registration.DEFAULT_MAX_PROCESSED_IMG_SIZE
        if has("micro_max_proc_size"):
            micro_max_proc_size = ei(namespace.micro_max_proc_size)

        if max_proc_size <= 0:
            raise ValueError(f"{max_proc_size=} <= 0")

        if micro_max_proc_size < max_proc_size:
            raise ValueError(f"{micro_max_proc_size=} < {max_proc_size=}")

        annotation_to_map_ids = []
        if has("annotation_to_map"):
            annotation_to_map_ids = eil(namespace.annotation_to_map)

        images_to_warp_ids = []
        if has("images_to_warp"):
            images_to_warp_ids = eil(namespace.images_to_warp)

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
        if not all_images_in_project:
            raise ValueError("unable to fetch all images")
        all_annotations_in_project = models.AnnotationCollection()
        all_annotations_in_project.project = project.id
        all_annotations_in_project.showWKT = True
        all_annotations_in_project.showTerm = True
        all_annotations_in_project.fetch()

        img_cache = {img.id: img for img in all_images_in_project}
        ann_cache = {ann.id: ann for ann in all_annotations_in_project}

        annotation_to_map = [ann_cache[id] for id in annotation_to_map_ids]
        if any(ann.image not in all_image_ids for ann in annotation_to_map):
            raise ValueError("annotations may only come from images in the registration sequence")

        return JobParameters(
            reference_image=img_cache.get(ref_image_id, None),
            all_images=[img_cache[idx] for idx in all_image_ids],
            image_ordering=image_ordering,
            align_toward_reference=align_toward_reference,
            image_crop=image_crop,
            registration_type=registration_type,
            compose_non_rigid=compose_non_rigid,
            max_proc_size=max_proc_size,
            micro_max_proc_size=micro_max_proc_size,
            annotations_to_map=annotation_to_map,
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
            self.update(19, "Downloaded images")

            self.update(20, "Creating registrar")
            # NOTE Valis doesn't allow 'lazy' folders: \
                # the images must be downloaded before creating the registrar
            registrar = registration.Valis(**self.get_valis_args())
            self.update(21, "Created registrar")

            self.update(22, "Registering all images")
            self.register(registrar)
            self.update(59, "Registered all images")

            self.update(60, "Warping all annotations")
            self.map_annotations_to_source_images(registrar, self.parameters.all_images)
            self.update(69, "Warped all annotations")

            self.update(70, "Warping images")
            img_lst = self.warp_images(registrar)

            if (
                self.parameters.map_annotations_to_warped_images
                and self.parameters.images_to_warp
            ):
                self.update(89, "Warped all images")
                self.update(90, "Warping annotations to uploaded images")
                self.map_annotations_to_uploaded_images(registrar, img_lst)
                self.update(99, "Warped all annotations to uploaded images")
            else:
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
            "max_image_dim_px": self.parameters.max_proc_size,
            "max_processed_image_dim_px": self.parameters.max_proc_size,
            "max_non_rigid_registartion_dim_px": self.parameters.max_proc_size,
        }

        # skip non rigid registrations
        if self.parameters.registration_type == RegistrationType.RIGID:
            valis_args["non_rigid_registrar_cls"] = None

        # reference image
        if self.parameters.reference_image is not None:
            valis_args["reference_img_f"] = str(
                self.src_dir / self.get_fname(self.parameters.reference_image)
            )

        return valis_args

    def get_fname(self, image: models.ImageInstance) -> str:
        if self.parameters.image_ordering == ImageOrdering.CREATED:
            return f"{image.created}_{image.filename}"
        return image.filename

    def get_thumb_path(self, image: models.ImageInstance) -> str:
        base = self.src_dir / self.get_fname(image)
        return str(base.with_suffix(".png"))

    def get_warped_image_path_ome_tiff(self, image: models.ImageInstance) -> str:
        fname = "{app}_{utc}_{name}".format(
            app=self.cytomine_job.job.id,
            utc=datetime.datetime.utcnow().strftime("%Y,%m,%d-%H,%M,%S"),
            name=image.filename,
        )
        return str((self.dst_dir / "saved-images" / fname).with_suffix(".ome.tiff"))

    def download_all_images(self):
        for image in self.parameters.all_images:
            img_path = self.get_thumb_path(image)
            bkp = image.filename
            try:
                status = image.download(img_path, override=False)
                image.filename = bkp
            except ValueError as e:
                raise ValueError(
                    f"could not download image {image.path} ({image.id}) "
                ) from e
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
            image_path = str(self.src_dir / self.get_fname(image))
            if image_path == registrar.reference_img_f:
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
        self.logger.info("reference image: %s", registrar.reference_img_f)

        if self.parameters.registration_type == RegistrationType.MICRO:
            with no_output():
                if self.parameters.micro_max_proc_size is not None:
                    micro_registrar, _ = registrar.register_micro(
                        max_non_rigid_registartion_dim_px=self.parameters.micro_max_proc_size
                    )
                else:
                    micro_registrar, _ = registrar.register_micro()

            self.logger.info("micro registration done")

        return rigid_registrar, non_rigid_registrar, micro_registrar

    def _find_image_group(self, images: typing.Sequence[models.ImageInstance]):
        "find an image group with all given images"
        images_ids = set(img.id for img in images)
        if not images_ids:
            return None

        # find all image groups for all images of the set
        image_groups_ids: typing.List[int] = []
        for img in images:
            ig_ii_c = models.ImageGroupImageInstanceCollection().fetch_with_filter(
                "imageinstance", img.id
            )
            if ig_ii_c is False:
                raise ValueError(f"unable to fetch image groups for {img=}")
            for ig_ii_ in ig_ii_c:
                image_groups_ids.append(ig_ii_.group)

        # find all images for each one of these groups (if any)
        for ig_ in image_groups_ids:
            ig_ii_c = models.ImageGroupImageInstanceCollection().fetch_with_filter(
                "imagegroup", ig_
            )
            if ig_ii_c is False:
                raise ValueError(f"unable to fetch image group {ig_=}")

            found_img_ids = set(ig_ii.image for ig_ii in ig_ii_c)

            # find the first one to include all images
            if found_img_ids.intersection(images_ids) == images_ids:
                image_group = models.ImageGroup().fetch(ig_)
                if image_group is False:
                    raise ValueError(f"unable to fetch image group {ig_}")
                return image_group

        return None

    def map_annotation_from_to(
        self,
        annotation: models.Annotation,
        src_image: models.ImageInstance,
        dst_image: models.ImageInstance,
        registrar: registration.Valis,
    ):
        src_slide: registration.Slide = registrar.get_slide(self.get_fname(src_image))
        dst_slide: registration.Slide = registrar.get_slide(self.get_fname(dst_image))

        src_geometry_bl = shapely.wkt.loads(annotation.location)
        src_geometry_tl = affine_transform(
            src_geometry_bl, [1, 0, 0, -1, 0, src_image.height]
        )

        # NOTE need to update when downloading at a lower resolution
        # src_shape = image_shape(self.get_thumb_path(src_image))
        # src_geometry_file_tl = affine_transform(src_geometry_tl,[src_shape[0] / \
            # src_image.width,0,0,src_shape[1] / src_image.height,0,0])
        src_geometry_file_tl = src_geometry_tl

        def warper_(x, y, z=None):
            assert z is None
            xy = np.stack([x, y], axis=1)
            warped_xy = src_slide.warp_xy_from_to(xy, dst_slide)

            return warped_xy[:, 0], warped_xy[:, 1]

        dst_geometry_file_tl = transform(warper_, src_geometry_file_tl)

        # NOTE need to update when downloading at a lower resolution
        # dst_shape = image_shape(self.get_thumb_path(dst_image))
        # dst_geometry_tl = affine_transform(dst_geometry_file_tl,[dst_image.width \
            # / dst_shape[0],0,0,dst_image.height / dst_shape[1],0,0])
        dst_geometry_tl = dst_geometry_file_tl
        dst_geometry_bl = affine_transform(
            dst_geometry_tl, [1, 0, 0, -1, 0, dst_image.height]
        )

        if not dst_geometry_bl.is_valid:
            raise ValueError(f"warping {annotation.id} produced an invalid geometry")

        return models.Annotation(
            shapely.wkt.dumps(dst_geometry_bl),
            dst_image.id,
            annotation.term,
            annotation.project,
        )

    def map_annotation_to_reference(
        self,
        annotation: models.Annotation,
        src_image: models.ImageInstance,
        dst_image: models.ImageInstance,
        registrar: registration.Valis,
    ):
        src_slide: registration.Slide = registrar.get_slide(self.get_fname(src_image))

        src_geometry_bl = shapely.wkt.loads(annotation.location)
        src_geometry_tl = affine_transform(
            src_geometry_bl, [1, 0, 0, -1, 0, src_image.height]
        )

        # NOTE need to update when downloading at a lower resolution
        # src_shape = image_shape(self.get_thumb_path(src_image))
        # src_geometry_file_tl = affine_transform(src_geometry_tl,[src_shape[0]\
            # / src_image.width,0,0,src_shape[1] / src_image.height,0,0])
        src_geometry_file_tl = src_geometry_tl

        def warper_(x, y, z=None):
            assert z is None
            xy = np.stack([x, y], axis=1)
            warped_xy = src_slide.warp_xy(xy)

            return warped_xy[:, 0], warped_xy[:, 1]

        dst_geometry_file_tl = transform(warper_, src_geometry_file_tl)

        # no need to scale up: this is the file that was uploaded
        dst_geometry_tl = dst_geometry_file_tl
        dst_geometry_bl = affine_transform(
            dst_geometry_tl, [1, 0, 0, -1, 0, dst_image.height]
        )

        if not dst_geometry_bl.is_valid:
            raise ValueError(f"warping {annotation.id} produced an invalid geometry")

        return models.Annotation(
            shapely.wkt.dumps(dst_geometry_bl),
            dst_image.id,
            annotation.term,
            annotation.project,
        )

    def _map_annotations_generic(
        self,
        registrar: registration.Valis,
        images: typing.Sequence[models.ImageInstance],
        mode: typing.Union[typing.Literal["to-warped"], typing.Literal["to-source"]],
        image_group: typing.Optional[models.ImageGroup] = None,
    ):
        if mode == "to-warped":
            mapper = self.map_annotation_to_reference
        elif mode == "to-source":
            mapper = self.map_annotation_from_to
        else:
            raise ValueError(f"invalid value for {mode=!r}")

        if not self.parameters.annotations_to_map:
            return
        if not images:
            return

        image_cache = {int(img.id): img for img in self.parameters.all_images}

        for annotation in self.parameters.annotations_to_map:
            src_img = image_cache[annotation.image]

            an_c = models.AnnotationCollection()
            for dst_image in images:
                an = mapper(annotation, src_img, dst_image, registrar)

                if not an.save():
                    self.logger.error("unable to save new annotation")
                an_c.append(an)

            if image_group is not None:
                ag = models.AnnotationGroup(src_img.project, image_group.id)
                if ag.save() is False:
                    raise ValueError("cannot create annotation group")
                for an in an_c:
                    al = models.AnnotationLink(
                        id_annotation=an.id, id_annotation_group=ag.id
                    )
                    if al.save() is False:
                        raise ValueError(f"could not link {an.id=!r}")

    def map_annotations_to_uploaded_images(
        self,
        registrar: registration.Valis,
        images: typing.Sequence[models.ImageInstance],
    ):
        image_group = self._find_image_group(images)
        assert image_group is not None, "an image group should have been created"
        self._map_annotations_generic(registrar, images, "to-warped", image_group)

    def map_annotations_to_source_images(
        self,
        registrar: registration.Valis,
        images: typing.Sequence[models.ImageInstance],
    ):
        image_group = self._find_image_group(images)
        self._map_annotations_generic(registrar, images, "to-source", image_group)

    def warp_images(self, registrar: registration.Valis):
        images: typing.List[models.ImageInstance] = []

        if not self.parameters.images_to_warp:
            return images

        # get storage
        userJob = models.UserJob().fetch(id=self.cytomine_job.job.userJob)
        all_storage = models.StorageCollection().fetch()
        if not all_storage:
            raise ValueError("cannot fetch storages for this project")
        user_storage = [s for s in all_storage if s.user == userJob.user][0]

        # warp all images and save to disk
        warped_images: typing.List[str] = []
        for image in self.parameters.images_to_warp:
            fname_src = self.get_fname(image)
            path_dst = self.get_warped_image_path_ome_tiff(image)

            slide: registration.Slide = registrar.get_slide(fname_src)
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

        images = self.get_image_instances(uploaded_files)
        if len(images) != len(uploaded_files):
            self.logger.error(
                "some images were not properly uploaded, no image group will be created"
            )
            return images

        ig_name = "{} {}".format(
            self.cytomine_job.software.name,
            datetime.datetime.utcnow().strftime("%Y/%m/%d %H:%M:%S"),
        )
        ig = models.ImageGroup(ig_name, self.cytomine_job.project.id).save()
        if ig is False:
            raise ValueError("unable to create a new image group")

        for image in images:
            if models.ImageGroupImageInstance(ig.id, image.id).save() is False:
                raise ValueError(f"unable to add {image.id=} to {ig.id=}")

        return images

    def get_image_instances(self, uploaded_files: typing.List[models.UploadedFile]):

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
            except KeyError as err:
                self.logger.exception(err)
                return None

        lst = retry(_get_lst)
        if lst is None:
            raise ValueError("impossible to fetch the list of images")

        if len(lst) == 0:
            raise ValueError("fetched empty list of images")

        return lst


def main(arguments):
    "starts the job with the given CLI arguments"
    with cytomine.CytomineJob.from_cli(arguments) as job:

        job.job.update(
            status=models.Job.RUNNING, progress=0, status_comment="Initialization"
        )

        src_dir = pathlib.Path(f"./valis-slides-{job.software.id}")
        dst_base_dir = pathlib.Path(f"./valis-results-{job.software.id}")

        dst_dir = dst_base_dir / "internal"
        name = "main"

        # check all parameters and fetch from Cytomine
        parameters = JobParameters.check(job.parameters, job.project)

        src_dir.mkdir(exist_ok=True, parents=False)
        dst_base_dir.mkdir(exist_ok=True, parents=False)

        logging.basicConfig()
        logger = logging.getLogger("cytomine.client")
        logger.setLevel(logging.DEBUG)

        VALISJob(job, parameters, src_dir, dst_dir, name, logger).run()

        job.job.update(
            status=models.Job.TERMINATED, progress=100, status_comment="Job terminated"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
