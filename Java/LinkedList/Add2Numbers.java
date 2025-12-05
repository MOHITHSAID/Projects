// Java file LinkedList contains the orginal implementation of Entire linked List
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if(l1==null) return l2;
        if(l2==null) return l1;
        ListNode dummy=new ListNode(-1);
        int carry=0;
        ListNode curr=dummy;
        int digit=0;
        while(l1!=null || l2!=null || carry!=0){
            int a;
            int b;
            if(l1!=null){
            a=l1.val;
            }
            else{
                a=0;
            }
            if(l2!=null){
            b=l2.val;
            }
            else{
                b=0;
            }
            int sum=a+b+carry;
            digit=sum%10;
            carry =sum/10;
            curr.next=new ListNode(digit);
            curr=curr.next;
            if (l1 != null) l1 = l1.next;
            if (l2 != null) l2 = l2.next;
        }
    return dummy.next;

    }

