"""ComfyUI VectorReady — vectorization-friendly post-processing for AI layer outputs.

Registers nodes under category 'VectorReady/*'. Loaded by ComfyUI via the
NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS convention."""

__version__ = "0.14.0"

from .nodes.alpha_cleanup import VR_AlphaCleanup
from .nodes.alpha_edge_refine import VR_AlphaEdgeRefine
from .nodes.alpha_stepify import VR_AlphaStepify
from .nodes.bilateral import VR_Bilateral
from .nodes.canny_edge import VR_CannyEdge
from .nodes.debug_probe import VR_DebugProbeImage, VR_DebugProbeMask, VR_SplitRGBA, vr_log
from .nodes.edge_aware_merge import VR_EdgeAwareMerge
from .nodes.edge_consistency_restore import VR_EdgeConsistencyRestore
from .nodes.empty_image_like import VR_EmptyImageLike
from .nodes.gated_passthrough import VR_GatedPassthrough
from .nodes.join_rgba import VR_JoinRGBA
from .nodes.kmeans_quantize import VR_KMeansQuantize
from .nodes.lab_convert import VR_LABConvert
from .nodes.hf_matting_alpha import VR_HFMattingAlpha
from .nodes.layer_matting_model_bridge import VR_LayerMattingModelBridge
from .nodes.layer_matting_refine import VR_LayerMattingRefine
from .nodes.layer_source_composer import VR_LayerSourceComposer
from .nodes.locate_anything_box import VR_LocateAnythingBox
from .nodes.mask_subtract import VR_MaskSubtract, VR_MaskUnion
from .nodes.reference_latent_if_mask_usable import VR_ReferenceLatentIfMaskUsable
from .nodes.roi_unsharp import VR_ROIUnsharpMask
from .nodes.target_mask_resolver import VR_TargetMaskResolver
from .nodes.target_trimap_builder import VR_TargetTrimapBuilder

vr_log("PLUGIN_VERSION", f"comfyui_vector_ready v{__version__}")
from .presets.pipeline import VR_PipelineLight, VR_PipelineStrong
from .presets.pipeline_debug import VR_PipelineLightDebug, VR_PipelineStrongDebug

NODE_CLASS_MAPPINGS = {
    "VR_LABConvert": VR_LABConvert,
    "VR_Bilateral": VR_Bilateral,
    "VR_KMeansQuantize": VR_KMeansQuantize,
    "VR_EdgeAwareMerge": VR_EdgeAwareMerge,
    "VR_EdgeConsistencyRestore": VR_EdgeConsistencyRestore,
    "VR_AlphaCleanup": VR_AlphaCleanup,
    "VR_AlphaEdgeRefine": VR_AlphaEdgeRefine,
    "VR_AlphaStepify": VR_AlphaStepify,
    "VR_ROIUnsharpMask": VR_ROIUnsharpMask,
    "VR_CannyEdge": VR_CannyEdge,
    "VR_EmptyImageLike": VR_EmptyImageLike,
    "VR_GatedPassthrough": VR_GatedPassthrough,
    "VR_JoinRGBA": VR_JoinRGBA,
    "VR_HFMattingAlpha": VR_HFMattingAlpha,
    "VR_LayerMattingModelBridge": VR_LayerMattingModelBridge,
    "VR_LayerMattingRefine": VR_LayerMattingRefine,
    "VR_LayerSourceComposer": VR_LayerSourceComposer,
    "VR_LocateAnythingBox": VR_LocateAnythingBox,
    "VR_MaskSubtract": VR_MaskSubtract,
    "VR_MaskUnion": VR_MaskUnion,
    "VR_ReferenceLatentIfMaskUsable": VR_ReferenceLatentIfMaskUsable,
    "VR_TargetMaskResolver": VR_TargetMaskResolver,
    "VR_TargetTrimapBuilder": VR_TargetTrimapBuilder,
    "VR_PipelineLight": VR_PipelineLight,
    "VR_PipelineStrong": VR_PipelineStrong,
    "VR_PipelineLightDebug": VR_PipelineLightDebug,
    "VR_PipelineStrongDebug": VR_PipelineStrongDebug,
    "VR_DebugProbeImage": VR_DebugProbeImage,
    "VR_DebugProbeMask": VR_DebugProbeMask,
    "VR_SplitRGBA": VR_SplitRGBA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VR_LABConvert": "VR · LAB Convert",
    "VR_Bilateral": "VR · Bilateral Filter",
    "VR_KMeansQuantize": "VR · K-means Quantize (LAB)",
    "VR_EdgeAwareMerge": "VR · Edge-aware Region Merge",
    "VR_EdgeConsistencyRestore": "VR · Edge Consistency Restore",
    "VR_AlphaCleanup": "VR · Alpha Cleanup (median + open/close)",
    "VR_AlphaEdgeRefine": "VR · Alpha Edge Refine",
    "VR_AlphaStepify": "VR · Alpha Stepify",
    "VR_ROIUnsharpMask": "VR · ROI Unsharp Mask",
    "VR_CannyEdge": "VR · Canny Edge",
    "VR_EmptyImageLike": "VR · Empty Image Like (size-matched solid color)",
    "VR_GatedPassthrough": "VR · Gated Passthrough (A/B switch)",
    "VR_JoinRGBA": "VR · Join RGBA (opacity convention, no inversion)",
    "VR_HFMattingAlpha": "VR · HF Matting Alpha",
    "VR_LayerMattingModelBridge": "VR · Layer Matting Model Bridge",
    "VR_LayerMattingRefine": "VR · Layer Matting Refine",
    "VR_LayerSourceComposer": "VR · Layer Source Composer",
    "VR_LocateAnythingBox": "VR · LocateAnything Box",
    "VR_MaskSubtract": "VR · Mask Subtract (outer − inner cutout)",
    "VR_MaskUnion": "VR · Mask Union (max of two masks)",
    "VR_ReferenceLatentIfMaskUsable": "VR · Reference Latent If Mask Usable",
    "VR_TargetMaskResolver": "VR · Target Mask Resolver",
    "VR_TargetTrimapBuilder": "VR · Target Trimap Builder",
    "VR_PipelineLight": "VR · Pipeline (Light, A-path)",
    "VR_PipelineStrong": "VR · Pipeline (Strong, B-path)",
    "VR_PipelineLightDebug": "VR · Pipeline Light (DEBUG, 26 outputs)",
    "VR_PipelineStrongDebug": "VR · Pipeline Strong (DEBUG, 11 outputs)",
    "VR_DebugProbeImage": "VR · Debug Probe (IMAGE)",
    "VR_DebugProbeMask": "VR · Debug Probe (MASK)",
    "VR_SplitRGBA": "VR · Split RGBA → RGB + Alpha",
}

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
