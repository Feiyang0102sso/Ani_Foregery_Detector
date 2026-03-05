def cut_file_lines(input_path, output_path, end_line):
    """
    截取文件第 0 行到 end_line 行（包含 end_line）并保存为新文件
    :param input_path: 原文件路径
    :param output_path: 输出文件路径
    :param end_line: 截取到的行号（从0开始）
    """
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 截取 0 到 end_line（包含）
    sliced = lines[:end_line + 1]

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(sliced)


if __name__ == "__main__":
    input_file = r"E:\!project dataset\data_list\val\0000.json"
    output_file = r"E:\!project dataset\data_list\val\0000_cc.json"
    line_number = 47385  # 截取到第100行（包含）

    cut_file_lines(input_file, output_file, line_number)
    print("文件截取完成")