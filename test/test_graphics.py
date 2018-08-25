#!/usr/bin/python

from graphics import *

def main():
    win = GraphWin("My Circle", 200, 200, autoflush=False)
    c1 = Circle(Point(50,50), 10)
    c1.setFill("red")
    c2 = Circle(Point(150,150), 10)
    c2.setFill("green")
    c1.draw(win)
    c2.draw(win)
    for i in range(100):
        c1.move(1, 1)
        c2.move(-1, -1)
        update(rate=10.0)
    win.getMouse() # Pause to view result
    win.close()    # Close window when done

if __name__ == "__main__":
    main()

