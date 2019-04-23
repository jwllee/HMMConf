package nl.tue.alignment.test;

public class BitShiftTest {
	public final static void main(String args[]) {
		// place 0 and place 1 have 1 token each
		byte block = (byte) 0b11000000;
		System.out.println("Block: " + block);
		// check that place 0 has 1 token
		byte checkPlace0 = (byte) 0b10000000;
		System.out.println((block & checkPlace0) != 0 );
		// check that place 2 has 1 token should give false
		byte checkPlace2 = (byte) 0b00100000;
		System.out.println((block & checkPlace2) != 0);
	}
}
