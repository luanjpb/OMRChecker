import cv2
import io
from pathlib import Path
from PIL import Image

from src.defaults import CONFIG_DEFAULTS
from src.template import Template
from src.utils.parsing import (
    get_concatenated_response, 
    template_with_defaults_from_dict,
    config_with_defaults_from_dict
)

def process_buffers(
    buffers,
    template
):
    files_counter = 0
    
    results = []

    for buffer in buffers:
        files_counter += 1
        file_name = buffer["name"]

        in_omr = cv2.imdecode(
            buffer["buffer"], cv2.IMREAD_GRAYSCALE
        )

        in_omr = template.image_instance_ops.apply_preprocessors(
            "", in_omr, template
        )

        if in_omr is None:
            results.append(
                {
                    "id": file_name,
                    "status": "error"
                }
            )
            continue

        # uniquify
        file_id = str(file_name)
        save_dir = None
        (
            response_dict,
            final_marked,
            multi_marked,
            _,
        ) = template.image_instance_ops.read_omr_response(
            template, image=in_omr, name=file_id, save_dir=save_dir
        )

        # TODO: move inner try catch here
        # concatenate roll nos, set unmarked responses, etc
        omr_response = get_concatenated_response(response_dict, template)

        marked_image = io.BytesIO()
        # image.save expects a file-like as a argument
        Image.fromarray(final_marked).save(marked_image, format="JPEG")

        if multi_marked == 0:
            results.append(
                {
                    "id": file_name,
                    "status": "success",
                    "marked_image": marked_image,
                    "answers": omr_response
                }
            )
        else:
            results.append(
                {
                    "id": file_name,
                    "status": "multi_marked",
                    "marked_image": marked_image,
                    "answers": omr_response
                }
            )

    return results


def process(
    buffers,
    config=None,
    template=None,
    tuning_config=CONFIG_DEFAULTS
):
    tuning_config = config_with_defaults_from_dict(
        config or {}
    )

    # Update local template (in current recursion stack)
    template = Template(
        Path("./inputs/template.json"),
        tuning_config,
        json_object=template_with_defaults_from_dict(
            template or {}
        )
    )
       
    return process_buffers(
        buffers,
        template
    )
