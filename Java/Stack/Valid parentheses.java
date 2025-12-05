class Solution {
    public boolean isValid(String s) {
        Stack<Character>s1=new Stack<Character>();
        for(char c:s.toCharArray()){
            if(c=='(' || c=='{' ||c=='['){
                s1.push(c);
            }
            else{
                if (s1.isEmpty()) return false;
               if(c==')' || c=='}' || c==']'){
                if((s1.peek()!='(' && c==')') || (s1.peek()!='{' && c=='}') || (s1.peek()!='[' && c==']')){
                    return false;
                }
               } 
            s1.pop();
            }
            
        }
        return s1.isEmpty();
    }
}
