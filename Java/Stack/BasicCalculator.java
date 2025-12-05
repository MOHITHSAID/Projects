class Solution {
    public int calculate(String s) {
        int res=0;
        int num=0;
        int sign=1;
        Stack<Integer>s1=new Stack<Integer>();
       for(char c:s.toCharArray()){
        if(Character.isDigit(c)){
            num=10*num+(int)(c-'0');
        }
        if(c=='+'){
            res=res+num*sign;
            num=0;
            sign=1;
        }
        if(c=='-'){
            res=res+num*sign;
            num=0;
            sign=-1;
        }
        if(c=='('){
            s1.push(res);
            s1.push(sign);
            sign=1;
            res=0;
        }
        else if(c==')'){
            res=res+sign*num;
            num=0;
            res=res*s1.pop();
            res=res+s1.pop();
        }
       }
       if(num!=0){
        res=res+num*sign;
       }
       return res;
    }
}
