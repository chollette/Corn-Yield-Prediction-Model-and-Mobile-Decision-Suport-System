
# This repo is built for paper: Corn Yield Prediction Model with Deep Neural Networks for Smallholder Farmer Decision Support System (...)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub Issues](https://img.shields.io/github/issues/yourusername/yourproject.svg)](https://github.com/yourusername/yourproject/issues)
[![GitHub Stars](https://img.shields.io/github/stars/yourusername/yourproject.svg)](https://github.com/yourusername/yourproject/stargazers)

## Description

Crop yield prediction has been modeled on the assumption that there is no interaction between weather and soil variables.  However, this paper argues that an interaction exists, and it can be finely modelled using the Kendall Correlation coefficient. Given the nonlinearity of the interaction between weather and soil variables, a deep neural network regressor (DNNR) is carefully designed with consideration to the depth, number of neurons of the hidden layers, and the hyperparameters with their optimizations. Additionally, a new metric, the average of absolute root squared error (ARSE) is proposed to combine the strengths of root mean square error (RMSE) and mean absolute error (MAE). With the ARSE metric, the proposed DNNR(s), optimised random forest regressor (RFR) and the extreme gradient boosting regressor (XGBR) achieved impressively small yield errors, 0.0172 t/ha, and 0.0243 t/ha, 0.0001 t/ha, and 0.001 t/ha, respectively. However, the DNNR(s), with changes to the explanatory variables to ensure generalizability to unforeseen data, DNNR(s) performed best. Further analysis reveals that a strong interaction does exist between weather and soil variables. Precisely, yield is observed to increase when precipitation is reduced and silt increased, and vice-versa. However, the degree of decrease or increase is not quantified in this paper. Contrary to existing yield models targeted towards agricultural policies and global food security, the goal of the proposed corn yield model is to empower the smallholder farmer to farm smartly and intelligently, thus the prediction model is integrated into a mobile application that includes education, and a farmer-to-market access module. 

## Deployment: Crop Yield Flask App
### Run with virtualenv
- Create a virtualenv folder `virtualenv -p python3 venv`
- Activate `source venv/bin/activate`
- Install the requirements `pip install -r requirements.txt`
- Run gunicorn `gunicorn app:app --bind 0.0.0.0:5000 --reload`

## Run with Docker
- Build the image with `docker build -t hello-app .`
- Run the container with `docker run -d -p 5000:5000 -e PORT=5000 --name hello-server hello-app`
- Check that the container is running
- Go to `localhost:5000`. It should output the yield prediction.


## Citation
If it is helpful to your work, please cite this paper:

@misc{title={Corn Yield Prediction Model with Deep Neural Networks for Smallholder Farmer Decision Support System}, 
      author={Chollette Olisah, Lyndon Smith, Melvyn Smith, Lawrence Morolake, Osi Ojukwu},
      year={2024},
      eprint={arXiv:2401.03768},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
}

## License

This project is licensed under the MIT License.

Chollette, Corn-Yield-Prediction-Model-and-Mobile-Decision-Suport-System
