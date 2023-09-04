import pandas as pd

from pipeline_forge.utils import (
    dict_params_to_int,
)

# Define test data and variables
sample_dict = {'param1': 1.5, 'param2': 2.7, 'param3': 3.1}
int_params = ['param1', 'param3']
sample_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4.0, 5.1, 6.2]})
sample_path = 'test_data.pkl'


def test_dict_params_to_int():
    modified_dict = dict_params_to_int(sample_dict, int_params)
    assert isinstance(modified_dict['param1'], int)
    assert isinstance(modified_dict['param3'], int)
    assert isinstance(modified_dict['param2'], float)
