from mazes.maze_one import MazeOne
from mazes.maze_two import MazeTwo


def main() -> None:
    """
    Выполнение задач.
    Активация двух лабиринтов.
    """

    # region ----------------------- Первый Лабиринт --------------------------------
    # Составной класс.
    maze_one = MazeOne()
    maze_one.execute()
    # endregion ---------------------------------------------------------------------

    # region ----------------------- Второй Лабиринт --------------------------------
    # Дочерний класс.
    maze_two = MazeTwo()
    maze_two.execute()
    # endregion ---------------------------------------------------------------------


if __name__ == '__main__':
    main()
