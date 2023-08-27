package interact;

public class main {
    public static void main(String[] args) {
        String content = String.format("name is: %s","MalGanis");
        System.out.println(content);
        System.out.println("What is that Thing?");
        frameWriter fw = new frameWriter();
        System.out.println(String.format("id is: %d",fw.getId()));

        Polygon poly = new Polygon();
        Point p1 = new Point(1.0,2.0);
        System.out.println(p1);

        System.out.println(poly.addEdge("Node1", "Node2"));
        System.out.println(poly.addEdge("Node2","not a Node"));
        Point p2 = new Point(2.0,1.0);

        System.out.println(p2 == p1);
        System.out.println(p1 == p2);
    }
}