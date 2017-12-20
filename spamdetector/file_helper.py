from os import path

class FileHelper:
    def exist_path(self, path_to_check:str) -> bool:
        """
        Verify if a path exist on the machine, returns a True in case it exist, otherwise False
        :param path_to_check:
        :return: boolean
        """
        return path.exists(path_to_check)

    def read_pattern_separated_file(self, file_path:str, pattern:str) -> list:
        """
        Read each of a text file and split them using the provided pattern

        :param file_path: File path to read
        :param pattern: The pattern used to split each line
        :return Array: The array that contain the splitted lines
        """
        pattern_separated_lines = []
        if file_path and self.exist_path(file_path):
            with open(file_path, encoding='UTF-8') as f:
                lines = f.readlines()
                pattern_separated_lines = [line.strip().split(pattern) for line in lines]

        return pattern_separated_lines
