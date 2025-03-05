# e2e_Gradutation_DNA_ds

## Overview

This repository contains the end-to-end project for DNA sequence analysis and paternity testing. The project involves data exploration, preparation, feature engineering, data augmentation, and model training. The goal is to develop a robust machine learning model to predict paternity based on DNA sequence data.

## Project Structure

The project is organized into several Jupyter notebooks, each focusing on a specific part of the data science workflow:

1. **00_Exploration.ipynb**: Initial data exploration and visualization.
2. **01_Preparation.ipynb**: Data cleaning and preprocessing.
3. **02_Feature_Engineering.ipynb**: Feature extraction and engineering.
4. **03_Augmentation.ipynb**: Data augmentation to address dataset limitations.
5. **04a_Model_Notebook_K3.ipynb**: Model training and evaluation (Kernel 3).
6. **04b_Model_Notebook_K6.ipynb**: Model training and evaluation (Kernel 6).
7. **04c_Model_Notebook_K7.ipynb**: Model training and evaluation (Kernel 7).
8. **04d_Model_Notebook_K8.ipynb**: Model training and evaluation (Kernel 8).
9. **server_k7_simulation.ipynb**: Simulation and testing of the model on server K7.

## Notebooks Overview

### 00_Exploration.ipynb

- Initial data exploration to understand the dataset.
- Visualization of DNA sequences and basic statistics.

### 01_Preparation.ipynb

- Data cleaning and preprocessing steps.
- Handling missing values and data normalization.

### 02_Feature_Engineering.ipynb

- Feature extraction using the Needleman-Wunsch algorithm for sequence alignment.
- Creation of target labels based on alignment scores.

### 03_Augmentation.ipynb

- Data augmentation to generate additional samples.
- Addressing dataset limitations and improving model performance.

### 04a_Model_Notebook_K3.ipynb to 04d_Model_Notebook_K8.ipynb

- Training and evaluation of different models using various kernels.
- Comparison of model performance and selection of the best model.

### server_k7_simulation.ipynb

- Simulation and testing of the trained model on server K7.
- Evaluation of model performance on real-world data.

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/e2e_Gradutation_DNA_ds.git
    ```
2. Navigate to the project directory:
    ```bash
    cd e2e_Gradutation_DNA_ds
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Open the Jupyter notebooks and run the cells to execute the code.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Special thanks to the contributors and the open-source community for their valuable tools and libraries.
- This project was developed as part of a graduation project in DNA sequence analysis.

For any questions or inquiries, please contact [mohamed.abdulaziz.eisa@gmail.com].
