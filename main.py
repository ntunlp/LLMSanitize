from lmsanitize import DataContaminationChecker


def main():
    dataset_name = "Rowan/hellaswag"
    contamination_checker = DataContaminationChecker(dataset_name, dataset_name)
    contamination_checker.run_contamination("gpt-2")

if __name__ == '__main__':
    main()