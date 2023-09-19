package interact;

public class Point {
    double x = 0.0;
    double y = 0.0;
    String label = "defaultPointLabel";

    public Point(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public String toString() {
        return String.format("Point: %.2f %.2f",this.x,this.y);
    }

    public boolean compare(Point other) {
        return this.x == other.x && this.y == other.y;
    }

}
