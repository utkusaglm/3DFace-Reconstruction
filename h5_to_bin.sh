h5dump -d /color/model/mean -b LE -o ./data/color_model_mean.bin model2017-1_face12_nomouth.h5
h5dump -d /color/model/pcaVariance -b LE -o ./data/color_model_pcaVariance.bin model2017-1_face12_nomouth.h5
h5dump -d /color/model/pcaBasis -b LE -o ./data/color_model_pcaBasis.bin model2017-1_face12_nomouth.h5

h5dump -d /expression/model/mean -b LE -o ./data/expression_model_mean.bin model2017-1_face12_nomouth.h5
h5dump -d /expression/model/pcaVariance -b LE -o ./data/expression_model_pcaVariance.bin model2017-1_face12_nomouth.h5
h5dump -d /expression/model/pcaBasis -b LE -o ./data/expression_model_pcaBasis.bin model2017-1_face12_nomouth.h5

h5dump -d /shape/model/mean -b LE -o ./data/shape_model_mean.bin model2017-1_face12_nomouth.h5
h5dump -d /shape/model/pcaVariance -b LE -o ./data/shape_model_pcaVariance.bin model2017-1_face12_nomouth.h5
h5dump -d /shape/model/pcaBasis -b LE -o ./data/shape_model_pcaBasis.bin model2017-1_face12_nomouth.h5
h5dump -d /shape/representer/cells -b LE -o ./data/shape_representer_cells.bin model2017-1_face12_nomouth.h5
