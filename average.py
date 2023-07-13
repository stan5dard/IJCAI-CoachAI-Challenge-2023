import csv
import os

def calculate_average_column_vectors(csv_files):
    # Create a list to store the column vectors
    column_vectors = []

    # Iterate over each CSV file
    for csv_file in csv_files:
        if not os.path.isfile(csv_file):
            print(f"File '{csv_file}' does not exist.")
            continue

        with open(csv_file, 'r') as file:
            reader = csv.reader(file)

            # Read the first row to get the headers
            headers = next(reader, None)

            # Read the remaining rows and store each column in a list
            for index, column in enumerate(zip(*reader)):
                # If the column vector doesn't exist yet, create a new list
                if len(column_vectors) <= index:
                    column_vectors.append([])

                # Append the values to the column vector
                column_vectors[index].extend(map(float, column))

    # Calculate the average for each column vector
    average_column_vector = [sum(column) / len(column) for column in column_vectors]

    return average_column_vector, headers

def save_column_vector_to_csv(column_vector, headers, output_file):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerow(column_vector)

    print(f"Average column vector saved to '{output_file}'.")

# Provide the list of CSV files
csv_files = ['./model_3_crossvalidation/lr0.0001bat32dim32alpha0.4layer1epoch300/prediction0.csv', 
             './model_3_crossvalidation/lr0.0001bat32dim32alpha0.4layer1epoch300/prediction1.csv', 
             './model_3_crossvalidation/lr0.0001bat32dim32alpha0.4layer1epoch300/prediction2.csv', 
             './model_3_crossvalidation/lr0.0001bat32dim32alpha0.4layer1epoch300/prediction3.csv', 
             './model_3_crossvalidation/lr0.0001bat32dim32alpha0.4layer1epoch300/prediction4.csv']


# Call the function to calculate the average column vector
average_vector, headers = calculate_average_column_vectors(csv_files)

# Provide the path to the output CSV file
output_file = './model_3_crossvalidation/lr0.0001bat32dim32alpha0.4layer1epoch300/average_column_vector.csv'

# Call the function to save the average column vector to a CSV file
save_column_vector_to_csv(average_vector, headers, output_file)
