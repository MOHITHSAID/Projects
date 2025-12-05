class Solution {
    public String removeKdigits(String num, int k) {
        Stack<Character>res=new Stack<Character>();
        for(char c:num.toCharArray()){
            while(!res.isEmpty() && (k>0) && (res.peek()>c)){
                res.pop();
                k--;
            }
            res.push(c);
        }
        while(!res.isEmpty() && k>0){
            res.pop();
            k--;
        }
        StringBuilder sb=new StringBuilder();
        for(char c:res){
            sb.append(c);
        }
        while((sb.length()>1)&&(sb.charAt(0)=='0')  ){
            sb.deleteCharAt(0);
        }
        if(sb.length()==0) return "0";
        return sb.toString();

    }
}
