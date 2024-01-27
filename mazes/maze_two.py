from typing import Optional

from PIL import Image, ImageDraw
import itertools


# region ---------------------------- БАЗОВЫЙ КЛАСС ---------------------------------
class _Maze:
    """Основа лабиринта."""
    def __init__(self) -> None:
        self._maze: list[list[int]] = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
        self._start: tuple[int, int] = (1, 1)
        self._end: tuple[int, int] = (5, 19)
# endregion -------------------------------------------------------------------------


# region ---------------------------- КЛАСС МАТРИЦЫ ---------------------------------
class _Matrix(_Maze):
    """Создание матрицы."""
    def __init__(self) -> None:
        super().__init__()
        self._matrix: list = []

    def _create_matrix(self) -> None:
        """Создание матрицы."""
        for i in range(len(self._maze)):
            self._matrix.append([])
            for j in range(len(self._maze[i])):
                self._matrix[-1].append(0)

    def _set_start_point(self) -> None:
        """Установить стартовую точку."""
        row, column = self._start
        self._matrix[row][column] = 1


class _MazeAnimation(_Matrix):
    """Анимировать лабиринт."""
    def __init__(self) -> None:
        super().__init__()
        self._images: list = []
        self._zoom: int = 20
        self._borders: int = 6

    def _draw_matrix(
            self,
            maze: list[list[int]],
            matrix: list[list[int]],
            the_path: Optional[list[tuple[int, int]]] = None,
    ) -> None:
        """Нарисовать матрицу."""
        if the_path is None:
            the_path = []

        im = Image.new(
            mode='RGB',
            size=(self._zoom * len(maze[0]), self._zoom * len(maze)),
            color=(255, 255, 255),
        )
        draw = ImageDraw.Draw(im)
        for i in range(len(maze)):
            for j in range(len(maze[i])):
                color = (255, 255, 255)
                r = 0

                if maze[i][j] == 1:
                    color = (0, 0, 0)

                if i == self._start[0] and j == self._start[1]:
                    color = (0, 255, 0)
                    r = self._borders

                if i == self._end[0] and j == self._end[1]:
                    color = (0, 255, 0)
                    r = self._borders

                draw.rectangle(
                    xy=(
                        j * self._zoom + r,
                        i * self._zoom + r,
                        j * self._zoom + self._zoom - r - 1,
                        i * self._zoom + self._zoom - r - 1,
                    ),
                    fill=color
                )

                if matrix[i][j] > 0:
                    r = self._borders
                    draw.ellipse(
                        xy=(
                            j * self._zoom + r,
                            i * self._zoom + r,
                            j * self._zoom + self._zoom - r - 1,
                            i * self._zoom + self._zoom - r - 1,
                        ),
                        fill=(255, 0, 0),
                    )

        for u in range(len(the_path) - 1):
            y = the_path[u][0] * self._zoom + int(self._zoom / 2)
            x = the_path[u][1] * self._zoom + int(self._zoom / 2)
            y1 = the_path[u + 1][0] * self._zoom + int(self._zoom / 2)
            x1 = the_path[u + 1][1] * self._zoom + int(self._zoom / 2)
            draw.line(xy=(x, y, x1, y1), fill=(255, 0, 0), width=5)
        draw.rectangle(
            xy=(0, 0, self._zoom * len(maze[0]), self._zoom * len(maze)),
            outline=(0, 255, 0),
            width=2
        )
        self._images.append(im)

    def _glimmer_of_way_out_maze(self, the_path: list[tuple[int, int]]):
        """Мерцание пути при окончании нахождении выхода."""
        for i in range(10):
            if i % 2 == 0:
                self._draw_matrix(self._maze, self._matrix, the_path)
            else:
                self._draw_matrix(self._maze, self._matrix)

    def _create_file_gif_of_maze(self):
        """Создать файл с гиф-изображением лабиринта."""
        self._images[0].save(
            'maze.gif',
            save_all=True,
            append_images=self._images[1:],
            optimize=False,
            duration=1,
            loop=0,
        )
# endregion -------------------------------------------------------------------------


# region ----------------------------- КЛАСС ПОИСКА ---------------------------------
class _MazeShortestWay(_MazeAnimation):
    """Поиск кратчайшего пути из лабиринта."""
    def __init__(self) -> None:
        super().__init__()
        self._the_path: Optional[list[tuple[int, int]]] = None

    def _finding_way(self):
        """Найти кратчайший путь."""
        row, column = self._end
        k = self._matrix[row][column]
        self._the_path = [(row, column)]
        while k > 1:
            if row > 0 and self._matrix[row - 1][column] == k - 1:
                row, column = row - 1, column
                self._the_path.append((row, column))
                k -= 1
            elif column > 0 and self._matrix[row][column - 1] == k - 1:
                row, column = row, column - 1
                self._the_path.append((row, column))
                k -= 1
            elif (
                    row < len(self._matrix) - 1 and
                    self._matrix[row + 1][column] == k - 1
            ):
                row, column = row + 1, column
                self._the_path.append((row, column))
                k -= 1
            elif (
                    column < len(self._matrix[row]) - 1 and
                    self._matrix[row][column + 1] == k - 1
            ):
                row, column = row, column + 1
                self._the_path.append((row, column))
                k -= 1
            self._draw_matrix(self._maze, self._matrix, self._the_path)
# endregion -------------------------------------------------------------------------


# region ----------------------------- КЛАСС ШАГОВ ----------------------------------
class _MazeMakeStep(_MazeShortestWay):
    """Шаги по лабиринту."""

    def __is_first_condition(self, row: int, column: int) -> bool:
        """
        Если текущая `строка` больше единицы и `верхняя` клетка пустая
        (на неё не ходили) и если она не является стеной, то => пойдем на неё.
        """
        return bool(
            row > 0 and
            self._matrix[row - 1][column] == 0 and
            self._maze[row - 1][column] == 0
        )

    def __is_second_condition(self, row: int, column: int) -> bool:
        """
        Если текущая позиция `столбца` больше единицы и `левая` клетка пустая
        (на неё не ходили) и если она не является стеной, то => пойдем на неё.
        """
        return bool(
            column > 0 and
            self._matrix[row][column - 1] == 0 and
            self._maze[row][column - 1] == 0
        )

    def __is_third_condition(self, row: int, column: int) -> bool:
        """
        Если текущая `строка` не больше лабиринта и `нижняя` клетка пустая
        (на неё не ходили) и если она не является стеной, то => пойдем на неё.
        """
        return bool(
            row < len(self._matrix) - 1 and
            self._matrix[row + 1][column] == 0 and
            self._maze[row + 1][column] == 0
        )

    def __is_fourth_condition(self, row: int, column: int) -> bool:
        """
        Если текущая позиция `столбца` не больше лабиринта и `правая` клетка пустая
        (на неё не ходили) и если она не является стеной, то => пойдем на неё.
        """
        return bool(
            column < len(self._matrix[row]) - 1 and
            self._matrix[row][column + 1] == 0 and
            self._maze[row][column + 1] == 0
        )

    def __make_step(self, step: int) -> None:
        """Сделать указанный шаг и обозначить его цифрой на матрице."""
        for row in range(len(self._matrix)):
            for column in range(len(self._matrix[row])):
                if self._matrix[row][column] == step:

                    if self.__is_first_condition(row=row, column=column):
                        self._matrix[row - 1][column] = step + 1

                    if self.__is_second_condition(row=row, column=column):
                        self._matrix[row][column - 1] = step + 1

                    if self.__is_third_condition(row=row, column=column):
                        self._matrix[row + 1][column] = step + 1

                    if self.__is_fourth_condition(row=row, column=column):
                        self._matrix[row][column + 1] = step + 1

    def _fill_space_with_dots_to_end(self) -> None:
        """Заполнить пространство цифрами ходов до выхода лабиринта и прерваться."""
        for step in itertools.count(start=1):
            if self._matrix[self._end[0]][self._end[1]] != 0:
                break
            self.__make_step(step)
            self._draw_matrix(self._maze, self._matrix)
# endregion -------------------------------------------------------------------------


# region ------------------------------ КЛАСС КОНСОЛИ -------------------------------
class _MazeDisplayInConsole(_MazeMakeStep):
    """Отобразить лабиринт на консоли."""

    def _print_m(self) -> None:
        """Вывести путь на консоль."""
        for i in range(len(self._matrix)):
            for j in range(len(self._matrix[i])):
                print(str(self._matrix[i][j]).ljust(2), end=' ')
            print()
# endregion -------------------------------------------------------------------------


class MazeTwo(_MazeDisplayInConsole,
              _MazeMakeStep,
              _MazeShortestWay,
              _MazeAnimation,
              _Matrix,
              _Maze):
    """Второй лабиринт."""

    def execute(self) -> None:
        """Выполнить создание лабиринта и его анимацию."""
        self._create_matrix()
        self._set_start_point()
        self._fill_space_with_dots_to_end()
        self._finding_way()
        self._glimmer_of_way_out_maze(self._the_path)
        self._print_m()
        print(self._the_path)
        self._create_file_gif_of_maze()
