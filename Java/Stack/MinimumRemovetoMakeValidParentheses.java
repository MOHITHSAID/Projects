class Solution {
    public String minRemoveToMakeValid(String s) {
        StringBuffer res=new StringBuffer();
        int balance=0;
        for(char c:s.toCharArray()){
            if(c=='('){
                balance=balance+1;
            }
            else if(c==')'){
                if(balance==0) {
                    continue;
                }
                else{
                    balance=balance-1;
                }
            }
            
            res.append(c);
        }
        int open=balance;
        StringBuffer res2=new StringBuffer();
        for(int i=res.length()-1;i>=0;i--){
            char c=res.charAt(i);
            if(balance>0 && c=='(') {
                balance--;
                continue;
            }
            res2.append(c);

        }
        return res2.reverse().toString();
    }
}
