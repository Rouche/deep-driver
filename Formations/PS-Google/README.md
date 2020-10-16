
### Resources

Labs
https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/introduction_to_tensorflow/labs

### Issues

Failed validating 'additionalProperties' in markdown_cell:

    On instance['cells'][124]:
    {'cell_type': 'markdown',
     'metadata': {},
     'outputs': ['...0 outputs...'],
     'source': '## Lifecycles, naming, and watching\n'
               '\n'
               'In Python-based TensorFlow,...'}
               
https://github.com/nteract/nteract/issues/1306

Restart Kernell and Clear Output in a new notebook, then save / reload the file.

Or clear the `outputs:` property on markdown cell type. 