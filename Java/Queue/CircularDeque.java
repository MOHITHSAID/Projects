class MyCircularDeque {

    int[] arr;
    int f, r, size;

    public MyCircularDeque(int k) {
        size = k;
        arr = new int[size];
        f = -1;
        r = -1;
    }

    public boolean insertFront(int value) {
        if (isFull()) return false;

        // First insertion
        if (isEmpty()) {
            f = r = 0;
        } else {
            f = (f - 1 + size) % size;
        }

        arr[f] = value;
        return true;
    }

    public boolean insertLast(int value) {
        if (isFull()) return false;

        // First insertion
        if (isEmpty()) {
            f = r = 0;
        } else {
            r = (r + 1) % size;
        }

        arr[r] = value;
        return true;
    }

    public boolean deleteFront() {
        if (isEmpty()) return false;

        if (f == r) {  // only one element
            f = r = -1;
        } else {
            f = (f + 1) % size;
        }

        return true;
    }

    public boolean deleteLast() {
        if (isEmpty()) return false;

        if (f == r) { // only one element
            f = r = -1;
        } else {
            r = (r - 1 + size) % size;
        }

        return true;
    }

    public int getFront() {
        return isEmpty() ? -1 : arr[f];
    }

    public int getRear() {
        return isEmpty() ? -1 : arr[r];
    }

    public boolean isEmpty() {
        return f == -1;
    }

    public boolean isFull() {
        return (f == 0 && r == size - 1) || (f == r + 1);
    }
}
