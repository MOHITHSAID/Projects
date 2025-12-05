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

    
     public Node swapNodes(int k) {
        Node n1=head;
        int length=0;
        while(n1!=null){
            n1=n1.next;
            length=length+1;
        }
        int arr[]=new int[length];
        int i=0;
        while(head!=null && i<length){
            arr[i]=head.data;
            head=head.next;
            i++;
        }
        int temp=arr[k-1];
        arr[k-1]=arr[length-k];
        arr[length-k]=temp;
        Node dummy=new Node(-1);
        Node res=dummy;
        for(int val:arr){
            Node n3=new Node(val);
            res.next=n3;
            res=res.next;
        }
        return dummy.next;

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
		LinkedList.head=LinkedList.swapNodes(2);
		LinkedList.print();
	}
}