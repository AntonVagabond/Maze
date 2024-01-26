from __future__ import annotations

from typing import TYPE_CHECKING, Optional, TypeAlias

from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import random
from queue import Queue

if TYPE_CHECKING:
    from matplotlib.lines import Line2D
    from numpy import ndarray
    from numpy import float64

FuncAnimation: TypeAlias = animation.FuncAnimation


class _Maze:
    """Основа лабиринта."""
    def __init__(
            self,
            maze: Optional[ndarray[ndarray[float64]]] = None,
            path: Optional[list[tuple[int, int]]] = None,
    ) -> None:
        # Создать сетку, заполненную стенами (Матрицу).
        self._maze = maze
        # Определить возможные направления.
        self._directions: list[tuple[int, int]] = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self._path: Optional[list[tuple[int, int]]] = path


class _MazeCreate(_Maze):
    """Создание лабиринта."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        # размер лабиринта
        self._dim = dim
        # Координаты по оси x и y
        self._x: int = 0
        self._y: int = 0
        # Инициализация стека начальной точки.
        self._stack: list[tuple[int, int]] = [(self._x, self._y)]

    def __create_matrix(self) -> None:
        """Создать сетку, заполненную стенами (Матрицу)."""
        self._maze = np.ones((self._dim * 2 + 1, self._dim * 2 + 1))

    def __determine_start_point(self) -> None:
        """Определяем начальную точку."""
        self._maze[2 * self._x + 1, 2 * self._y + 1] = 0

    def __change_original_directions(self):
        """Изменить начальные направления."""
        random.shuffle(self._directions)

    def __establishing_paths(self) -> None:
        """Установление корректного пути и его создание у лабиринта."""

        for dx, dy in self._directions:
            nx, ny = self._x + dx, self._y + dy
            if (
                    0 <= nx < self._dim and
                    0 <= ny < self._dim and
                    self._maze[2 * nx + 1, 2 * ny + 1] == 1
            ):
                self._maze[2 * nx + 1, 2 * ny + 1] = 0
                self._maze[2 * self._x + 1 + dx, 2 * self._y + 1 + dy] = 0
                self._stack.append((nx, ny))
                break
        else:
            self._stack.pop()

    def __create_entry_and_exit(self) -> None:
        """Создание входа и выхода."""
        self._maze[1, 0] = 0
        self._maze[-2, -1] = 0

    def execute_create(self) -> ndarray[ndarray[float64]]:
        """Выполнить создание лабиринта."""
        self.__create_matrix()
        self.__determine_start_point()
        # Создаем пути у лабиринта.
        while len(self._stack) > 0:
            self._x, self._y = self._stack[-1]
            self._directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            self.__change_original_directions()
            self.__establishing_paths()

        self.__create_entry_and_exit()
        return self._maze


class _MazeFindingOut(_Maze):
    """Поиск кратчайшего выхода из лабиринта."""
    def __init__(self, maze: ndarray[ndarray[float64]]) -> None:
        super().__init__(maze=maze)
        self._visited: Optional[ndarray[ndarray[bool]]] = None
        self._start: tuple[int, int] = (1, 1)
        self._end: tuple[int, int] = (
            self._maze.shape[0] - 2, self._maze.shape[1] - 2
        )
        self._queue: Queue = Queue()
        self._node: Optional[tuple[int, int]] = None

    def __create_paths_from_booleans(self) -> None:
        """
        Создать пути из булевых значений.
        С помощью них будем проверять в каких местах лабиринта мы были.
        """
        self._visited = np.zeros_like(self._maze, dtype=bool)

    def __set_start_point_as_verified(self) -> None:
        """Установить начальную точку как проверенную."""
        self._visited[self._start] = True

    def __finding_best_way_out_maze(self) -> list[tuple[int, int]]:
        """Поиск наилучшего выхода из лабиринта."""
        while not self._queue.empty():
            self._node, self._path = self._queue.get()

            for dx, dy in self._directions:
                next_node = (self._node[0] + dx, self._node[1] + dy)

                if next_node == self._end:
                    return self._path + [next_node]

                if (
                        self._maze.shape[0] > next_node[0] >= 0 ==
                        self._maze[next_node] and
                        0 <= next_node[1] < self._maze.shape[1] and
                        not self._visited[next_node]
                ):
                    self._visited[next_node] = True
                    self._queue.put((next_node, self._path + [next_node]))

    def execute_find(self) -> list[tuple[int, int]]:
        """Выполнить нахождение кратчайшего пути."""
        # Алгоритм `Поиска в ширину` для нахождения кратчайшего пути.
        self.__create_paths_from_booleans()
        self.__set_start_point_as_verified()

        self._queue.put((self._start, []))
        path = self.__finding_best_way_out_maze()
        return path


class _DrawingExitFromMaze(_Maze):
    """Создать анимацию прохождения лабиринта кротчайшим путем."""
    def __init__(
            self,
            maze: ndarray[ndarray[float64]],
            path: Optional[list[tuple[int, int]]]
    ) -> None:
        super().__init__(maze=maze, path=path)
        # Создать макет, где будет изображен лабиринт.
        self._fig, self._ax = plt.subplots(figsize=(10, 10))
        self._line: tuple[Line2D]

    def __set_borders_in_white(self) -> None:
        """Установить границы `белым` цветом."""
        self._fig.patch.set_edgecolor('white')

    def __set_line_width(self) -> None:
        """Установить ширину линии."""
        self._fig.patch.set_linewidth(0)

    def __display_data_as_image(self) -> None:
        """Отображать данные в виде изображения."""
        self._ax.imshow(self._maze, cmap=plt.cm.binary, interpolation='nearest')

    def __remember_path(self) -> None:
        """Запомнить путь, и при отображении изобразить её красной линией."""
        line, = self._ax.plot([], [], color='red', linewidth=2)
        self._line = line

    def __init(self) -> tuple[Line2D]:
        """Функция инициализации."""
        self._line.set_data([], [])
        return self._line,

    def __update(self, frame: int) -> tuple[Line2D]:
        """Функция обновления."""
        self._line.set_data(*zip(*[(p[1], p[0]) for p in self._path[:frame + 1]]))
        return self._line,

    def __display_path_with_red_line(self) -> FuncAnimation:
        """Отобразить путь красной линией."""
        ani = animation.FuncAnimation(
            fig=self._fig,
            func=self.__update,
            frames=np.arange(len(self._path)),
            init_func=self.__init,
            blit=True,
            repeat=False,
            interval=20,
        )
        return ani

    def __draw_entry_and_exit_arrows(self) -> None:
        """Нарисовать стрелки входа и выхода."""
        # Отобразить стрелку входа.
        self._ax.arrow(
            x=0,
            y=1,
            dx=0.4,
            dy=0,
            fc='green',
            ec='green',
            head_width=0.3,
            head_length=0.3,
        )
        # Отобразить стрелку выхода.
        self._ax.arrow(
            x=self._maze.shape[1] - 1,
            y=self._maze.shape[0] - 2,
            dx=0.4,
            dy=0,
            fc='blue',
            ec='blue',
            head_width=0.3,
            head_length=0.3,
        )

    @staticmethod
    def __activate_animation() -> None:
        """Активировать анимацию."""
        plt.show()

    def execute_animate(self) -> None:
        """Выполнить анимацию кратчайшего пути."""
        self.__set_borders_in_white()
        self.__set_line_width()
        self.__display_data_as_image()

        if self._path is not None:
            self.__remember_path()
            ani = self.__display_path_with_red_line()

        self.__draw_entry_and_exit_arrows()
        self.__activate_animation()


class MazeOne:
    """Первый лабиринт."""

    def __init__(self) -> None:
        self._maze_create = _MazeCreate
        self._maze_finding_out = _MazeFindingOut
        self._drawing_exit_from_maze = _DrawingExitFromMaze

    def execute(self) -> None:
        """Выполнить создание -> поиск -> анимацию лабиринта."""
        # Создание лабиринта.
        maze_create = self._maze_create(dim=int(input('Введите размер лабиринта: ')))
        maze = maze_create.execute_create()
        # Нахождение кратчайшего пути.
        maze_finding_out = self._maze_finding_out(maze=maze)
        path = maze_finding_out.execute_find()
        # Отобразить лабиринт.
        drawing_exit_from_maze = self._drawing_exit_from_maze(maze=maze, path=path)
        drawing_exit_from_maze.execute_animate()
