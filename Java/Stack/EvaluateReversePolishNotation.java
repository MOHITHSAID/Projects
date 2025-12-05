class Solution {
    public int evalRPN(String[] tokens) {
        Stack<Integer> st=new Stack<Integer>();
        
        for(int i=0;i<tokens.length;i++){
            String token=tokens[i];
            switch(token){
                case "+":
                    st.push(st.pop()+st.pop());
                    break;
                case "-":
                    int a=st.pop();
                    int b=st.pop();
                    st.push(b-a);
                    break;
                case "*":
                    int A=st.pop();
                    int B=st.pop();
                    st.push(B*A);
                    break;
                case "/":
                    int a1=st.pop();
                    int b1=st.pop();
                    st.push(b1/a1);
                    break;
               default:
                    st.push(Integer.parseInt(token));
                    break;
            }
