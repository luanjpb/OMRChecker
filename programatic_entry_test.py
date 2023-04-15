from src.programatic_entry import process
import numpy as np
import json

if __name__ == "__main__":
    with open('./inputs/template.json') as file:
        template = json.load(file)

    with open('./inputs/teste/teste.jpeg', 'rb') as file:
        results = process(
            buffers=[{
                "name": "teste",
                "buffer": np.frombuffer(file.read(), np.uint8)
            }],
            template=template
        )

    print(f"{results=}")
