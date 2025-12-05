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

    
    public void rearrangeOddEven() {
        if (head==null||head.next==null) return;

        Node odd=head;           // points to 1st node
        Node even=head.next;     // points to 2nd node
        Node evenHead=even;      // to reconnect later

        while (even!=null && even.next!=null) {
            odd.next=even.next;      // skip even -> link odd to next odd
            odd=odd.next;

            even.next=odd.next;      // skip odd -> link even to next even
            even=even.next;
        }

        odd.next=evenHead;  // attach even list after odd list
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
		LinkedList.rearrangeOddEven();
		LinkedList.print();
	}
}