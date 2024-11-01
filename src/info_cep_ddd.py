import json
from pathlib import Path

# Function to save the combined data
def save_combined_data(file_path, data):
    with file_path.open("w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

# Main function
def main():
    # Combined structure with 'CEP', 'DDD', and 'REGIAO' for each state
    state_data = {
        'SP': {'CEP': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 'DDD': [11, 12, 13, 14, 15, 16, 17, 18, 19], 'REGIAO': ['Sudeste']},
        'RJ': {'CEP': [20, 21, 22, 23, 24, 25, 26, 27, 28], 'DDD': [21, 22, 24], 'REGIAO': ['Sudeste']},
        'ES': {'CEP': [29], 'DDD': [27, 28], 'REGIAO': ['Sudeste']},
        'MG': {'CEP': [30, 31, 32, 33, 34, 35, 36, 37, 38, 39], 'DDD': [31, 32, 33, 34, 35, 37, 38], 'REGIAO': ['Sudeste']},
        'PR': {'CEP': [80, 81, 82, 83, 84, 85, 86, 87], 'DDD': [41, 42, 43, 44, 45, 46], 'REGIAO': ['Sul']},
        'SC': {'CEP': [88, 89], 'DDD': [47, 48, 49], 'REGIAO': ['Sul']},
        'RS': {'CEP': [90, 91, 92, 93, 94, 95, 96, 97, 98, 99], 'DDD': [51, 53, 54, 55], 'REGIAO': ['Sul']},
        'DF': {'CEP': [70, 71, 72, 73], 'DDD': [61], 'REGIAO': ['Centro-Oeste']},
        'GO': {'CEP': [74, 75], 'DDD': [62, 64], 'REGIAO': ['Centro-Oeste']},
        'MT': {'CEP': [78], 'DDD': [65, 66], 'REGIAO': ['Centro-Oeste']},
        'MS': {'CEP': [79], 'DDD': [67], 'REGIAO': ['Centro-Oeste']},
        'TO': {'CEP': [77], 'DDD': [63], 'REGIAO': ['Norte']},
        'AC': {'CEP': [69], 'DDD': [68], 'REGIAO': ['Norte']},
        'RO': {'CEP': [76], 'DDD': [69], 'REGIAO': ['Norte']},
        'PA': {'CEP': [66, 67], 'DDD': [91, 93, 94], 'REGIAO': ['Norte']},
        'AM': {'CEP': [69], 'DDD': [92, 97], 'REGIAO': ['Norte']},
        'AP': {'CEP': [68], 'DDD': [96], 'REGIAO': ['Norte']},
        'RR': {'CEP': [69], 'DDD': [95], 'REGIAO': ['Norte']},
        'MA': {'CEP': [65], 'DDD': [98, 99], 'REGIAO': ['Nordeste']},
        'PI': {'CEP': [64], 'DDD': [86, 89], 'REGIAO': ['Nordeste']},
        'CE': {'CEP': [60, 61, 62, 63], 'DDD': [85, 88], 'REGIAO': ['Nordeste']},
        'RN': {'CEP': [59], 'DDD': [84], 'REGIAO': ['Nordeste']},
        'PB': {'CEP': [58], 'DDD': [83], 'REGIAO': ['Nordeste']},
        'PE': {'CEP': [50, 51, 52, 53, 54, 55, 56], 'DDD': [81, 87], 'REGIAO': ['Nordeste']},
        'AL': {'CEP': [57], 'DDD': [82], 'REGIAO': ['Nordeste']},
        'SE': {'CEP': [49], 'DDD': [79], 'REGIAO': ['Nordeste']},
        'BA': {'CEP': [40, 41, 42, 43, 44, 45, 46, 47, 48], 'DDD': [71, 73, 74, 75, 77], 'REGIAO': ['Nordeste']}
    }

    # Define the file path using Path
    base_folder = Path.cwd() / "src"
    base_folder.mkdir(exist_ok=True)  # Create the folder if it doesn't exist
    file_path = base_folder / "state_region_info.json"

    # Save the combined data
    save_combined_data(file_path, state_data)

if __name__ == "__main__":
    main()

