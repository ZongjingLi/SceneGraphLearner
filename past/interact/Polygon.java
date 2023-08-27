package interact;
import java.util.ArrayList;


public class Polygon {

    private ArrayList<String> labels = new ArrayList<String>();
    private ArrayList<Point> vertices = new ArrayList<Point>();
    private ArrayList<Integer> edges = new ArrayList<Integer>(); 
    
    public Polygon() {
        System.out.println("A Default Polygon is Created.");
        this.labels.add("Node1");
        this.labels.add("Node2");
        this.labels.add("Node3");
        System.out.println(String.format("Size of Label %d", this.labels.size()));
    }

    int getNodeNum() {return this.labels.size();}

    int getEdgeNum() {return this.edges.size();}

    ArrayList<String> getLabels() {return this.labels;}

    ArrayList<Point> getVertices() {return this.vertices;}

    ArrayList<Integer> getEdges() {return this.edges;}

    boolean addVertex(double x, double y) {
        this.vertices.add(new Point(x, y));
        return true;
    }

    boolean addEdge(String u, String v) {
        boolean hasU = hasNode(u);
        boolean hasV = hasNode(v);
        if (!hasU || !hasV) {return false;}

        return true;
    }

    boolean hasEdge(String u, String v) {
        return false;
    }

    boolean hasNode(String label) {
        boolean hasLabel = false;
        int size = this.getNodeNum();
        for(int i=0; i<size; i++) {
            String currLabel = this.labels.get(i);
            if (currLabel.equals(label) && !hasLabel) {
                hasLabel = true;
            }
        }
        if (hasLabel) {return true;}
        return false;
    }
}
