import imageio.v3 as imageio
import napari
from disentangle.core.tiff_reader import load_tiff
from micro_sam import instance_segmentation, util
from micro_sam.multi_dimensional_segmentation import automatic_3d_segmentation


def cell_segmentation():
    """Run the instance segmentation functionality from micro_sam for segmentation of
    HeLA cells. You need to run examples/annotator_2d.py:hela_2d_annotator once before
    running this script so that all required data is downloaded and pre-computed.
    """
    image_path = "/home/ashesh.ashesh/code/Disentangle/disentangle/notebooks/test_img.tiff"
    embedding_path = "../embeddings/embeddings-hela2d.zarr"

    # Load the image, the SAM Model, and the pre-computed embeddings.
    image = load_tiff(image_path)
    predictor = util.get_sam_model()
    embeddings = util.precompute_image_embeddings(predictor, image, save_path=embedding_path)

    # Use the instance segmentation logic of Segment Anything.
    # This works by covering the image with a grid of points, getting the masks for all the poitns
    # and only keeping the plausible ones (according to the model predictions).
    # While the functionality here does the same as the implementation from Segment Anything,
    # we enable changing the hyperparameters, e.g. 'pred_iou_thresh', without recomputing masks and embeddings,
    # to support (interactive) evaluation of different hyperparameters.

    # Create the automatic mask generator class.
    amg = instance_segmentation.AutomaticMaskGenerator(predictor)

    # Initialize the mask generator with the image and the pre-computed embeddings.
    amg.initialize(image, embeddings, verbose=True)

    # Generate the instance segmentation. You can call this again for different values of 'pred_iou_thresh'
    # without having to call initialize again.
    instances = amg.generate(pred_iou_thresh=0.88)
    instances = instance_segmentation.mask_data_to_segmentation(
        instances, shape=image.shape, with_background=True
    )

    # Show the results.
    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(instances)
    napari.run()


def cell_segmentation_with_tiling():
    """Run the instance segmentation functionality from micro_sam for segmentation of
    cells in a large image. You need to run examples/annotator_2d.py:wholeslide_annotator once before
    running this script so that all required data is downloaded and pre-computed.
    """
    image_path = "../data/whole-slide-example-image.tif"
    embedding_path = "../embeddings/whole-slide-embeddings.zarr"

    # Load the image, the SAM Model, and the pre-computed embeddings.
    image = imageio.imread(image_path)
    predictor = util.get_sam_model()
    embeddings = util.precompute_image_embeddings(
        predictor, image, save_path=embedding_path, tile_shape=(1024, 1024), halo=(256, 256)
    )

    # Use the instance segmentation logic of Segment Anything.
    # This works by covering the image with a grid of points, getting the masks for all the poitns
    # and only keeping the plausible ones (according to the model predictions).
    # The functionality here is similar to the instance segmentation in Segment Anything,
    # but uses the pre-computed tiled embeddings.

    # Create the automatic mask generator class.
    amg = instance_segmentation.TiledAutomaticMaskGenerator(predictor)

    # Initialize the mask generator with the image and the pre-computed embeddings.
    amg.initialize(image, embeddings, verbose=True)

    # Generate the instance segmentation. You can call this again for different values of 'pred_iou_thresh'
    # without having to call initialize again.
    instances = amg.generate(pred_iou_thresh=0.88)
    instances = instance_segmentation.mask_data_to_segmentation(
        instances, shape=image.shape, with_background=True
    )

    # Show the results.
    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(instances)
    v.add_labels(instances)
    napari.run()

if __name__ == "__main__":
    # Uncomment one of the two lines below to run the corresponding example.
    cell_segmentation()
    # cell_segmentation_with_tiling()