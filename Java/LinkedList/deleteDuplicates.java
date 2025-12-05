//LinkedList implemenation is in the file Linked List
public Node deleteDuplicates() {
        Node dummy=new Node(-1);
        dummy.next=head;
        Node prev=dummy;
        while(head!=null){
            if((head.next!=null)&&(head.data==head.data.val)){
                int val=head.data;
                while((head!=null)&&(head.data==data)){
                    head=head.next;
                }
                prev.next=head;
            }
            else{
                prev=prev.next;
                head=head.next;
                

            }
        }
        return dummy.next;
    }
}
