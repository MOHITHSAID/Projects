public Node removeElements(int val) {
        if(head==null) return null;
        Node n1=head;
        while(n1.next!=null){
            if(n1.next.data==data){
                n1.next=n1.next.next;
            }
           else{
             n1=n1.next;
           } 
           
        }
        if(head!=null && head.data==data) {
            head=head.next;
            }
    return head;
    }
