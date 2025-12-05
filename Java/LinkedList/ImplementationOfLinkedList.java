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
		LinkedList.print();
	}
}
