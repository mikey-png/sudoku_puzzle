import copy
import cv2
import numpy as np
import pytesseract
import re
import matplotlib.pyplot as plt
from typing import List

# 设置 Tesseract 路径
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        self.backtracking(board)

    def backtracking(self, board: List[List[str]]) -> bool:
        # 若有解，返回True；若无解，返回False
        for i in range(len(board)):  # 遍历行
            for j in range(len(board[0])):  # 遍历列
                # 若空格内已有数字，跳过
                if board[i][j] != '.':
                    continue
                for k in range(1, 10):  # 尝试数字1-9
                    if self.is_valid(i, j, k, board):
                        board[i][j] = str(k)  # 填入数字
                        if self.backtracking(board):  # 递归求解
                            return True
                        board[i][j] = '.'  # 回溯，撤回
                return False  # 若数字1-9都不能成功填入空格，返回False无解
        return True  # 有解

    def is_valid(self, row: int, col: int, val: int, board: List[List[str]]) -> bool:
        # 判断同一行是否冲突
        for i in range(9):
            if board[row][i] == str(val):
                return False
        # 判断同一列是否冲突
        for j in range(9):
            if board[j][col] == str(val):
                return False
        # 判断同一九宫格是否有冲突
        start_row = (row // 3) * 3
        start_col = (col // 3) * 3
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if board[i][j] == str(val):
                    return False
        return True


# 图像处理函数：从图片中提取数独
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges, img


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_grid_contour(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    return max_contour


def four_point_transform(image, pts):
    pts = pts.astype("float32")
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    warped = cv2.flip(warped, 1)
    warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return warped


def extract_numbers_from_grid(image):
    numbers = []
    h, w = image.shape[:2]
    cell_size = w // 9
    for i in range(9):
        row = []
        for j in range(9):
            x1, y1 = j * cell_size, i * cell_size
            x2, y2 = (j + 1) * cell_size, (i + 1) * cell_size
            cell = image[y1:y2, x1:x2]
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            _, cell = cv2.threshold(cell, 127, 255, cv2.THRESH_BINARY)
            cell = cv2.medianBlur(cell, 3)
            text = pytesseract.image_to_string(cell, config='--psm 10 --oem 3 --dpi 300')
            text = text.strip()
            if text == 'rs)':  # 如果识别错误，将其修改为 5
                text = '5'
            if re.match(r'^\d+$', text):
                row.append(text)
            else:
                row.append(".")
        numbers.append(row)
    return numbers


def convert_image_to_board(image_path):
    edges, original_img = preprocess_image(image_path)
    grid_contour = find_grid_contour(edges)
    rect_pts = cv2.approxPolyDP(grid_contour, 0.02 * cv2.arcLength(grid_contour, True), True)
    if len(rect_pts) == 4:
        transformed_img = four_point_transform(original_img, rect_pts.reshape(4, 2))
    else:
        raise ValueError("没有找到有效的四个角点")
    board = extract_numbers_from_grid(transformed_img)
    return board


# 绘制数独题目与答案
def plot_sudoku(board, solution):
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制9x9的方格
    ax.set_xticks(np.arange(0, 10, 1))
    ax.set_yticks(np.arange(0, 10, 1))
    ax.grid(which='both')

    # 设置刻度
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # 填充题目和答案
    for i in range(9):
        for j in range(9):
            # 题目数字以黑色显示
            if board[i][j] != ".":
                ax.text(j + 0.5, 8.5 - i, board[i][j], ha='center', va='center', fontsize=18, color='black')
            # 答案数字以红色显示，且只有空白的格子才显示答案
            elif solution[i][j] != ".":
                ax.text(j + 0.5, 8.5 - i, solution[i][j], ha='center', va='center', fontsize=18, color='red')

    # 显示图片
    # plt.title("Sudoku Puzzle and Solution")
    plt.show()


# 主程序示例
image_path = './aa.jpg'  # 数独图片路径
board = convert_image_to_board(image_path)

solution = copy.deepcopy(board)

# 使用 Solution 类求解数独
solution_class = Solution()
solution_class.solveSudoku(solution)

print("题目：")
for row in board:
    print(row)

print("答案：")
for row in solution:
    print(row)

# 绘制题目与答案
plot_sudoku(board, solution)