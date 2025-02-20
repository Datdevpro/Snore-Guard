
#Test model with user-provided audio file
from test_model import test_model
result = test_model('model/snoring_detector_model_0.99_2.h5', 'test_sound/long_duration_test.wav')
print(result)