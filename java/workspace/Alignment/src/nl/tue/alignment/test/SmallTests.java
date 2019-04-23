package nl.tue.alignment.test;

import java.util.Arrays;

import lpsolve.LpSolve;
import nl.tue.alignment.Progress;
import nl.tue.alignment.Utils;
import nl.tue.alignment.algorithms.ReplayAlgorithm;
import nl.tue.alignment.algorithms.ReplayAlgorithm.Debug;
import nl.tue.alignment.algorithms.implementations.AStar;
import nl.tue.alignment.algorithms.implementations.AStarLargeLP;
import nl.tue.alignment.algorithms.implementations.Dijkstra;
import nl.tue.alignment.algorithms.syncproduct.SyncProduct;
import nl.tue.alignment.algorithms.syncproduct.SyncProductImpl;
import nl.tue.astar.util.ilp.LPMatrixException;

public class SmallTests {

	public static byte LM = SyncProduct.LOG_MOVE;
	public static byte MM = SyncProduct.MODEL_MOVE;
	public static byte SM = SyncProduct.SYNC_MOVE;
	public static byte TM = SyncProduct.TAU_MOVE;
	public static int[] NE = SyncProduct.NOEVENT;
	public static int NR = SyncProduct.NORANK;

	public static class SyncProductExampleBook extends SyncProductImpl {

		public SyncProductExampleBook() {
			super("Book Example", 10,
					new String[] { "As,-", "Aa,-", "Fa,-", "Sso,-", "Ro,-", "Co,-", "t,-", "Da1,-", "Do,-", "Da2,-",
							"Ao,-", "Aaa,-", "As,As", "Aa,Aa", "Sso,Sso", "Ro,Ro", "Ao,Ao", "Aaa,Aaa1", "Aaa,Aaa2",
							"-,As", "-,Aa", "-,Sso", "-,Ro", "-,Ao", "-,Aaa1", "-,Aaa2" }, //
					new String[] { "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12",
							"p13", "p14", "p15", "p16", "p17", "p18" }, //
					new int[][] { NE, NE, NE, NE, NE, NE, NE, NE, NE, NE, NE, NE, { 0 }, { 1 }, { 2 }, { 3 }, { 4 },
							{ 5 }, { 6 }, { 0 }, { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 } }, //
					new int[] { NR, NR, NR, NR, NR, NR, NR, NR, NR, NR, NR, NR, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5,
							6 }, //
					new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, //
					new byte[] { MM, MM, MM, MM, MM, MM, TM, MM, MM, MM, MM, MM, SM, SM, SM, SM, SM, SM, SM, LM, LM, LM,
							LM, LM, LM, LM }, //
					//					new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, /**/ 19, 20, 21, 22, 23, 24, 24, /**/ 12, 13, 14,
					//							15, 16, 17, 17 }, //
					new int[] { 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1 }//
			);
			setInput(0, 0);
			setInput(1, 1);
			setInput(2, 2);
			setInput(3, 3);
			setInput(4, 4);
			setInput(5, 6);
			setInput(6, 5, 6);
			setInput(7, 1);
			setInput(8, 7);
			setInput(9, 8);
			setInput(10, 7);
			setInput(11, 9);
			//SyncMoves
			setInput(12, 0, 11);
			setInput(13, 1, 12);
			setInput(14, 3, 13);
			setInput(15, 4, 14);
			setInput(16, 7, 15);
			setInput(17, 9, 16);
			setInput(18, 9, 17);
			//LogMoves
			setInput(19, 11);
			setInput(20, 12);
			setInput(21, 13);
			setInput(22, 14);
			setInput(23, 15);
			setInput(24, 16);
			setInput(25, 17);

			setOutput(0, 1);
			setOutput(1, 2, 3);
			setOutput(2, 5);
			setOutput(3, 4);
			setOutput(4, 6);
			setOutput(5, 3);
			setOutput(6, 7);
			setOutput(7, 10);
			setOutput(8, 8);
			setOutput(9, 10);
			setOutput(10, 9);
			setOutput(11, 10);
			setOutput(12, 1, 12);
			setOutput(13, 2, 3, 13);
			setOutput(14, 4, 14);
			setOutput(15, 6, 15);
			setOutput(16, 9, 16);
			setOutput(17, 10, 17);
			setOutput(18, 10, 18);
			setOutput(19, 12);
			setOutput(20, 13);
			setOutput(21, 14);
			setOutput(22, 15);
			setOutput(23, 16);
			setOutput(24, 17);
			setOutput(25, 18);

			setInitialMarking(0, 11);
			setFinalMarking(10, 18);
		}

		public boolean isFinalMarking(byte[] marking) {
			// for full alignments:
			return Arrays.equals(marking, finalMarking);

			// for prefix alignments:
			// check only if place 18 marked with a single token
			//			return (marking[18 / 8] & (Utils.FLAG >>> (18 % 8))) != 0
			//					&& (marking[bm + 18 / 8] & (Utils.FLAG >>> (18 % 8))) == 0;
		}

	}

	public static class NastySyncProductExample extends SyncProductImpl {

		public NastySyncProductExample() {
			super("Nasty Example", 12,
					new String[] { "A,-", "B,-", "C,-", "D,-", "E,-", "F,-", "G,-", "H,-", "I,-", "J,-", "K,-", "L,-",
							"K,K", "L,L", "-,L", "-,K" }, //
					new String[] { "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12",
							"p13", "p14" }, //
					new int[][] { NE, NE, NE, NE, NE, NE, NE, NE, NE, NE, NE, NE, { 1 }, { 0 }, { 0 }, { 1 } }, //
					new int[] { NR, NR, NR, NR, NR, NR, NR, NR, NR, NR, NR, NR, 1, 0, 0, 1 }, //
					new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, //
					new byte[] { MM, MM, MM, MM, MM, MM, MM, MM, MM, MM, MM, MM, SM, SM, LM, LM }, //
					//					new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, /**/ 34, 35, /**/ 23, 22 }, //
					new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1 }//
			);
			setInput(0, 0);
			setOutput(0, 1, 2);

			setInput(1, 1);
			setOutput(1, 3);

			setInput(2, 3);
			setOutput(2, 7);

			setInput(3, 1);
			setOutput(3, 4);

			setInput(4, 4);
			setOutput(4, 7);

			setInput(5, 2);
			setOutput(5, 5);

			setInput(6, 5);
			setOutput(6, 8);

			setInput(7, 2);
			setOutput(7, 6);

			setInput(8, 6);
			setOutput(8, 8);

			setInput(9, 7, 8);
			setOutput(9, 9);

			setInput(10, 9);
			setOutput(10, 10);

			setInput(11, 10);
			setOutput(11, 11);

			setInput(12, 9, 13);
			setOutput(12, 10, 14);

			setInput(13, 10, 12);
			setOutput(13, 11, 13);

			setInput(14, 12);
			setOutput(14, 13);

			setInput(15, 13);
			setOutput(15, 14);

			setInitialMarking(0, 12);
			setFinalMarking(11, 14);
		}
	}

	//	public static class LoopExample extends SyncProductImpl {
	//
	//		public LoopExample() {
	//			super("Loop Example",
	//					new String[] { "A,-", "D,-", "C,-", "-,G", "-,C", "-,D", "C,C", "D,D", "F,-", "E,-", "-,E", "-,F",
	//							"E,E", "F,F", "G,-", "G,G", "-,X" }, //
	//					new String[] { "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12" }, //
	//					new int[] { NR, NR, NR, 1, 2, 3, 2, 3, NR, NR, 4, 5, 4, 5, NR, 1, 0 }, //
	//					new byte[] { MM, MM, MM, LM, LM, LM, SM, SM, MM, MM, LM, LM, SM, SM, MM, SM, LM }, //
	//					new int[] { 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1 }//
	//			);
	//
	//			setInput(0, 0);
	//			setOutput(0, 1);
	//
	//			setInput(1, 1);
	//			setOutput(1, 2);
	//
	//			setInput(2, 2);
	//			setOutput(2, 3);
	//
	//			setInput(3, 12);
	//			setOutput(3, 5);
	//
	//			setInput(16, 4);
	//			setOutput(16, 12);
	//
	//			setInput(4, 5);
	//			setOutput(4, 6);
	//
	//			setInput(5, 6);
	//			setOutput(5, 7);
	//
	//			setInput(6, 2, 5);
	//			setOutput(6, 3, 6);
	//
	//			setInput(7, 1, 6);
	//			setOutput(7, 2, 7);
	//
	//			setInput(8, 3);
	//			setOutput(8, 8);
	//
	//			setInput(9, 8);
	//			setOutput(9, 9);
	//
	//			setInput(10, 7);
	//			setOutput(10, 10);
	//
	//			setInput(11, 10);
	//			setOutput(11, 11);
	//
	//			setInput(12, 8, 7);
	//			setOutput(12, 9, 10);
	//
	//			setInput(13, 3, 10);
	//			setOutput(13, 8, 11);
	//
	//			setInput(14, 8);
	//			setOutput(14, 8);
	//
	//			setInput(15, 12, 8);
	//			setOutput(15, 5, 8);
	//
	//			setInitialMarking(0, 4);
	//			setFinalMarking(9, 11);
	//		}
	//	}
	//
	//	public static class LoopExample2 extends SyncProductImpl {
	//
	//		public LoopExample2() {
	//			super("Loop Example 2",
	//					new String[] { "A,-", "D,-", "C,-", "-,G", "-,C", "-,D", "C,C", "D,D", "F,-", "E,-", "-,E", "-,F",
	//							"E,E", "F,F", "G,-", "G,G", "tau" }, //
	//					new String[] { "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12" }, //
	//					new int[] { NR, NR, NR, 0, 1, 2, 1, 2, NR, NR, 3, 4, 3, 4, NR, 0, -1 }, //
	//					new byte[] { MM, MM, MM, LM, LM, LM, SM, SM, MM, MM, LM, LM, SM, SM, MM, SM, TM }, //
	//					new int[] { 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0 }//
	//			);
	//			setInput(0, 0);
	//			setOutput(0, 1);
	//
	//			setInput(1, 1);
	//			setOutput(1, 2);
	//
	//			setInput(2, 2);
	//			setOutput(2, 3);
	//
	//			setInput(3, 4);
	//			setOutput(3, 5);
	//
	//			setInput(4, 5);
	//			setOutput(4, 6);
	//
	//			setInput(5, 6);
	//			setOutput(5, 7);
	//
	//			setInput(6, 2, 5);
	//			setOutput(6, 3, 6);
	//
	//			setInput(7, 1, 6);
	//			setOutput(7, 2, 7);
	//
	//			setInput(8, 3);
	//			setOutput(8, 8);
	//
	//			setInput(9, 8);
	//			setOutput(9, 9);
	//
	//			setInput(10, 7);
	//			setOutput(10, 10);
	//
	//			setInput(11, 10);
	//			setOutput(11, 11);
	//
	//			setInput(12, 8, 7);
	//			setOutput(12, 9, 10);
	//
	//			setInput(13, 3, 10);
	//			setOutput(13, 8, 11);
	//
	//			setInput(14, 8);
	//			setOutput(14, 12);
	//
	//			setInput(16, 12);
	//			setOutput(16, 8);
	//
	//			setInput(15, 4, 8);
	//			setOutput(15, 5, 12);
	//
	//			setInitialMarking(0, 4);
	//			setFinalMarking(9, 11);
	//		}
	//	}
	//
	//	public static class TwoSwapsExample extends SyncProductImpl {
	//
	//		public TwoSwapsExample() {
	//			super("Two Swaps Example",
	//					new String[] { "A,-", "D,-", "C,-", "-,B", "-,C", "-,D", "C,C", "D,D", "F,-", "E,-", "-,E", "-,F",
	//							"E,E", "F,F", "-,G" }, //
	//					new String[] { "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12" }, //
	//					new int[] { NR, NR, NR, 0, 1, 2, 1, 2, NR, NR, 3, 4, 3, 4, 5 }, //
	//					new byte[] { MM, MM, MM, LM, LM, LM, SM, SM, MM, MM, LM, LM, SM, SM, LM }, //
	//					new int[] { 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1 }//
	//			);
	//			setInput(0, 0);
	//			setOutput(0, 1);
	//
	//			setInput(1, 1);
	//			setOutput(1, 2);
	//
	//			setInput(2, 2);
	//			setOutput(2, 3);
	//
	//			setInput(3, 4);
	//			setOutput(3, 5);
	//
	//			setInput(4, 5);
	//			setOutput(4, 6);
	//
	//			setInput(5, 6);
	//			setOutput(5, 7);
	//
	//			setInput(6, 2, 5);
	//			setOutput(6, 3, 6);
	//
	//			setInput(7, 1, 6);
	//			setOutput(7, 2, 7);
	//
	//			setInput(8, 3);
	//			setOutput(8, 8);
	//
	//			setInput(9, 8);
	//			setOutput(9, 9);
	//
	//			setInput(10, 7);
	//			setOutput(10, 10);
	//
	//			setInput(11, 10);
	//			setOutput(11, 11);
	//
	//			setInput(12, 8, 7);
	//			setOutput(12, 9, 10);
	//
	//			setInput(13, 3, 10);
	//			setOutput(13, 8, 11);
	//
	//			setInput(14, 11);
	//			setOutput(14, 12);
	//
	//			setInitialMarking(0, 4);
	//			setFinalMarking(9, 12);
	//		}
	//	}
	//
	public static void main(String[] args) throws LPMatrixException {
		// INITIALIZE LpSolve for stdout
		LpSolve.lpSolveVersion();

		int[] alignment;
		//		SyncProduct net = new SyncProductExampleBook();
		SyncProduct net = new NastySyncProductExample();
		Utils.toDot(net, System.out);

		alignment = testSingleGraph(net, Debug.DOT);
		//		Utils.toDot(net, System.out);
		if (alignment != null) {
			Utils.toDot(net, alignment, System.out);
		}
		//		testSingleGraph(new NastySyncProductExample(), Debug.DOT);
	}

	public static int[] testSingleGraph(SyncProduct net, Debug debug) throws LPMatrixException {

		ReplayAlgorithm algorithm;
		//INITIALIZATION OF CLASSLOADER FOR PROPER RECORDING OF TIMES.
		algorithm = new Dijkstra(net);
		algorithm = new AStarLargeLP(net);
		algorithm = new AStar(net);

		boolean dijkstra = false;
		boolean split = true;
		boolean moveSort = true; // moveSort on total order
		boolean queueSort = true; // queue sorted "depth-first"
		boolean preferExact = true; // prefer Exact solution
		//		int multiThread = 1; // do multithreading
		boolean useInt = false; //  use Integer

		if (dijkstra) {
			algorithm = new Dijkstra(net, //
					moveSort, // moveSort on total order
					queueSort, // queue sorted "depth-first"
					debug //
			);
		} else if (split) {
			algorithm = new AStarLargeLP(net, //
					moveSort, // moveSort on total order
					useInt, // use Integers
					debug // debug mode
			);

		} else {
			algorithm = new AStar(net, //
					moveSort, // moveSort on total order
					queueSort, // queue sorted "depth-first"
					preferExact, // prefer Exact solution
					useInt, // use Integers
					//					multiThread, // multithreading
					debug // debug mode
			);
		}

		try {
			return algorithm.run(Progress.INVISIBLE, Integer.MAX_VALUE, Integer.MAX_VALUE, Integer.MAX_VALUE);
		} catch (LPMatrixException e) {
			e.printStackTrace();
		}
		return null;
		//		for (int t : alignment) {
		//			System.out.println(net.getTransitionLabel(t));
		//		}
	}

}
