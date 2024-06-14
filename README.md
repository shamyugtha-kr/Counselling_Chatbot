
## Steps to Run the Project

1. **Download the Dataset**:
   - Place `train.csv`, `test.csv`, and `validation.csv` in the `data` directory.

2. **Preprocess the Data**:
    ```sh
    python preprocess.py
    ```

3. **Train the Model**:
    ```sh
    python models/train.py
    ```

4. **Run the Flask Application**:
    ```sh
    python main.py
    ```

## Requirements

- Python 3.8+
- Flask
- transformers
- torch
- pandas
