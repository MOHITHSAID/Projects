    public Node mergeTwoLists(Node list1,Node list2) {
        Node res=new Node(-1);
        Node ans=res;
        while(list1!=null && list2!=null){
            if(list1.data<=list2.data){
                Node n2=new Node(list1.data);
                res.next=n2;
                list1=list1.next;
            }
            else{
               Node n2=new Node(list2.data);
               res.next=n2;
               list2=list2.next;
            }
            res=res.next;
        }
    if(list1!=null){
        res.next=list1;
    }
    if(list2!=null){
        res.next=list2;
    }
    return ans.next;
    }
}
