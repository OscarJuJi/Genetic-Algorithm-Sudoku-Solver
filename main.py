import tkinter as tk
from interfaz import SudokuBoardGUI

if __name__ == "__main__":
    """
    Punto de entrada principal del programa.

    Este bloque se ejecuta si el script se está corriendo como programa principal (no importado como módulo).
    Crea una instancia de la ventana principal de Tkinter y la clase SudokuBoardGUI, y luego inicia el bucle de eventos de la interfaz gráfica.
    """
    root = tk.Tk()
    app = SudokuBoardGUI(root)

    app.run()
