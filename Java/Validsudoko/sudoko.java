class Solution {
    public boolean isValidSudoku(char[][] board) {
        HashSet<Character>[] rows=new HashSet[9];
        HashSet<Character>[] columns=new HashSet[9];
        HashSet<Character>[] box=new HashSet[9];
        for(int i=0;i<9;i++){          
            rows[i]=new HashSet<>();
            columns[i]=new HashSet<>();
            box[i]=new HashSet<>();
        }
        for(int i=0;i<9;i++){
            for(int j=0;j<9;j++){
                char num=board[i][j];

                if(num=='.') continue;               
                int boardindex=(i/3)*3+(j/3);
                if(rows[i].contains(num)) return false;
                if(columns[j].contains(num)) return false;
                if(box[boardindex].contains(num)) return false;
                rows[i].add(num);
                columns[j].add(num);
                box[boardindex].add(num);
            }
        }
        return true;

    }
}
