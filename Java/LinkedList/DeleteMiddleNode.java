// Node structure
class Node {
    int data;
    Node next;

    Node(int data) {
        this.data = data;
        this.next = null;
    }
}

// Linked List implementation
class MyLinkedList {
    Node head;

    // Insert at end (simple)
    public void insert(int data) {
        Node newNode=new Node(data);
        if (head==null) {
            head=newNode;
            return;
        }
        Node temp=head;
        while(temp.next != null) temp=temp.next;
        temp.next=newNode;
    }

    
     public Node deleteMiddle() {
        if(head==null||head.next==null) return null;

        Node n1=head;
        int length=0;
        while(n1!=null){
            length++;
            n1=n1.next;
        }
        Node n2=head;
        for(int i=0;i<(length/2)-1;i++){
            n2=n2.next;
        }
        n2.next=n2.next.next;
        return head;
    }


    // Print the linked list
    public void print() {
        Node temp=head;
        while(temp !=null) {
            System.out.print(temp.data + " -> ");
            temp=temp.next;
        }
        System.out.println("null");
    }
}
public class Main
{
	public static void main(String[] args) {
		MyLinkedList LinkedList=new MyLinkedList();
		LinkedList.insert(1);
		LinkedList.insert(2);
		LinkedList.insert(3);
		LinkedList.insert(4);
		LinkedList.insert(5);
		LinkedList.insert(6);
		LinkedList.deleteMiddle();
		LinkedList.print();
	}
}