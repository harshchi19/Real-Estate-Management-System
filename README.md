# Real Estate Price Prediction - CampusX

### Features

- **Price Prediction** system utilizing multiple machine learning models for various property types, including _Residential Apartments, Rental Apartments, Independent Floors, Independent Houses, Residential Houses_.
- Models include **Random Forest**, **CatBoost**, **HistGradientBoosting**, **LightGBM**, and **Gradient Boosting**. The best-performing model is **Gradient Boosting**, with an accuracy of 94.4%.
- **Web scraping** from [99acres.com](https://99acres.com) was done to gather real estate data.
- **Analytics Page** that offers insights into real estate trends for specific cities or localities in India.
- Option to **Add a New City** to make predictions and generate analytics based on the newly added city’s data.
- Ability to **Download Resources** such as datasets and trained machine learning models.

### Tech Stack

|                 Tech | Stack                       |
| -------------------: | :-------------------------- |
| Programming Language | Python                      |
|      Version Control | Git & GitHub                |
|        Data Analysis | Pandas, Numpy               |
|        Visualization | Matplotlib, Seaborn, Plotly |
|     Machine Learning | Scikit-Learn, XGBoost, CatBoost, LightGBM |
|   Frontend & Backend | Streamlit                   |
|                Extra | Pydantic                    |

### Installation

1. Clone this repository.
2. **Create a virtual environment** and install the required dependencies:

```sh
pip install -r requirements.txt
```

3. Run the Streamlit app:

```sh
streamlit run Real_Estate_Project.py
```

### Acknowledgements

- [99acres.com](https://99acres.com): Data source for this project.
- [@arv-anshul/99acres-scrape](https://github.com/arv-anshul/99acres-scrape): Used to scrape data from 99acres.com.
- [CampusX DSMP](https://learnwith.campusx.com): Inspiration and reference for the capstone project.

### License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).
