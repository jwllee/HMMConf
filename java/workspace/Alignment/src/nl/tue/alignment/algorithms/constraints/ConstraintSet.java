package nl.tue.alignment.algorithms.constraints;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import org.deckfour.xes.classification.XEventClass;
import org.deckfour.xes.classification.XEventClasses;
import org.processmining.models.graphbased.directed.petrinet.Petrinet;
import org.processmining.models.graphbased.directed.petrinet.PetrinetEdge;
import org.processmining.models.graphbased.directed.petrinet.PetrinetNode;
import org.processmining.models.graphbased.directed.petrinet.elements.Place;
import org.processmining.models.graphbased.directed.petrinet.elements.Transition;
import org.processmining.models.semantics.petrinet.Marking;
import org.processmining.plugins.connectionfactories.logpetrinet.TransEvClassMapping;

import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TObjectIntHashMap;

public class ConstraintSet {

	private TIntObjectMap<Set<Constraint>> label2input = new TIntObjectHashMap<>();
	private TIntObjectMap<Set<Constraint>> label2output = new TIntObjectHashMap<>();
	private Set<Constraint> constraints = new HashSet<>();
	private String[] colNames;

	private static final String TAU = "tau";
	private static final String VISIBLE = "??";

	public ConstraintSet(Petrinet net, Marking initialMarking, XEventClasses classes, TObjectIntMap<XEventClass> c2id,
			TransEvClassMapping map) {

		int ts = net.getTransitions().size();
		int ps = net.getPlaces().size();
		int cs = c2id.size();

		int rows = cs + ts + ps;
		int columns = ts + cs + 1;

		int[][] matrix = new int[rows][columns];
		int[] firstNonZero = new int[rows];
		Arrays.fill(firstNonZero, columns);
		colNames = new String[columns];
		colNames[columns - 1] = "marking";

		TObjectIntMap<Place> p2id = new TObjectIntHashMap<>(net.getPlaces().size(), 0.7f, -1);
		TObjectIntMap<Transition> t2id = new TObjectIntHashMap<>(net.getTransitions().size(), 0.7f, -1);

		for (Transition t : net.getTransitions()) {
			t2id.put(t, t2id.size());
		}

		for (PetrinetEdge<? extends PetrinetNode, ? extends PetrinetNode> edge : net.getEdges()) {
			if (edge.getSource() instanceof Place) {
				int p = p2id.putIfAbsent((Place) edge.getSource(), p2id.size());
				if (p < 0) {
					p = (p2id.size() - 1);
				}
				int t = t2id.putIfAbsent((Transition) edge.getTarget(), t2id.size());
				if (t < 0) {
					t = (t2id.size() - 1);
				}
				XEventClass clazz = map.get(edge.getTarget());
				int c;
				if (clazz == null) {
					c = -1;
					if (((Transition) edge.getTarget()).isInvisible()) {
						colNames[t] = TAU;
					} else {
						colNames[t] = VISIBLE;
					}
				} else {
					c = c2id.get(clazz);
				}

				// p --> t[c]
				if (c >= 0) {
					// t is mapped to c.
					matrix[c][t] = 1;
					matrix[c][ts + c] = -1;
					firstNonZero[c] = firstNonZero[c] > t ? t : firstNonZero[c];
					// t occurs less than c
					matrix[cs + t][t] = -1;
					matrix[cs + t][ts + c] = 1;
					firstNonZero[cs + t] = firstNonZero[cs + t] > t ? t : firstNonZero[cs + t];
					colNames[t] = clazz.toString().replace("+complete", "");
					colNames[ts + c] = clazz.toString().replace("+complete", "");
				}
				// t consumes from p
				matrix[cs + ts + p][t] -= 1;
				// initial marking
				matrix[cs + ts + p][ts + cs] = -initialMarking.occurrences(edge.getSource());
				firstNonZero[cs + ts + p] = firstNonZero[cs + ts + p] > t ? t : firstNonZero[cs + ts + p];

			} else {
				int p = p2id.putIfAbsent((Place) edge.getTarget(), p2id.size());
				if (p < 0) {
					p = (p2id.size() - 1);
				}
				int t = t2id.putIfAbsent((Transition) edge.getSource(), t2id.size());
				if (t < 0) {
					t = (t2id.size() - 1);
				}
				XEventClass clazz = map.get(edge.getSource());
				int c;
				if (clazz == null) {
					c = -1;
					if (((Transition) edge.getSource()).isInvisible()) {
						colNames[t] = TAU;
					} else {
						colNames[t] = VISIBLE;
					}
				} else {
					c = c2id.get(clazz);
				}

				if (c >= 0) {
					// t is mapped to c.
					matrix[c][t] = 1;
					matrix[c][ts + c] = -1;
					firstNonZero[c] = firstNonZero[c] > t ? t : firstNonZero[c];
					// t occurs less than c
					matrix[cs + t][t] = -1;
					matrix[cs + t][ts + c] = 1;
					firstNonZero[cs + t] = firstNonZero[cs + t] > t ? t : firstNonZero[cs + t];
					colNames[t] = clazz.toString().replace("+complete", "");
					colNames[ts + c] = clazz.toString().replace("+complete", "");
				}
				// t produces in p
				matrix[cs + ts + p][t] += 1;
				// initial marking
				matrix[cs + ts + p][ts + cs] = -initialMarking.occurrences(edge.getTarget());
				firstNonZero[cs + ts + p] = firstNonZero[cs + ts + p] > t ? t : firstNonZero[cs + ts + p];

			}

		}

		//		System.out.println("Before:");
		//		printMatrix(matrix);

		// swap rows cs+ts..cs+ts+ps to make the matrix triangular.
		for (int r = cs + ts; r < rows; r++) {
			for (int r2 = r + 1; r2 < rows; r2++) {
				if (firstNonZero[r2] < firstNonZero[r] || (firstNonZero[r2] == firstNonZero[r]
						&& firstNonZero[r] < columns && matrix[r2][firstNonZero[r]] < matrix[r][firstNonZero[r]])) {
					// swap rows
					int[] row = matrix[r2];
					matrix[r2] = matrix[r];
					matrix[r] = row;
					int first = firstNonZero[r2];
					firstNonZero[r2] = firstNonZero[r];
					firstNonZero[r] = first;
				}
			}
		}
		//		System.out.println("Sorted:");
		//		printMatrix(matrix);

		int[] first1 = new int[cs];
		for (int r = 0; r < cs; r++) {
			first1[r] = ts;
			for (int c = 0; c < ts && first1[r] == ts; c++) {
				if (matrix[r][c] == 1) {
					first1[r] = c;
				}
			}
		}

		int lastStrong = ps - 1;
		boolean done;
		// now matrix needs to be swept to create 0 columns in the lower left part.
		for (int c = 0; c < ts; c++) {

			int[][] newMatrix = new int[ps][];

			//			System.out.println("Column " + c);
			// try to reduce the lower elements of column c to 0, by
			// 1) subtracting or adding rows 0..cs-1
			// 2) adding a row from cs+ts..rows-1
			// 3) adding rows cs..cs+ts-1
			// without introducing non-zero elements in earlier columns
			for (int r = cs + ts; r < rows; r++) {
				// copy original
				int rn = r - cs - ts;
				newMatrix[rn] = Arrays.copyOf(matrix[r], matrix[r].length);

				done = matrix[r][c] == 0;
				//				if (!done) {
				//					System.out.println("Row " + r);
				//				}
				if (matrix[r][c] > 0) {

					if (colNames[c] == VISIBLE) {
						// mapped transition that has no corresponding event in the log.
						// set value of 0. 
						matrix[r][c] = 0;
						done = true;
					}

					// element at row r > 0
					// reduce by subtracting and element from row 0..cs-1
					for (int s = 0; s < cs && !done; s++) {
						if (first1[s] == c) {
							// subtract this row
							int f1 = matrix[r][c];
							for (int x = c; x < columns; x++) {
								newMatrix[rn][x] -= f1 * matrix[s][x];
							}
							done = true;
						}
					}

					// if not done, try to add another constraint
					for (int s = cs + ts; s < rows && !done; s++) {
						if (matrix[s][c] < 0) {
							int f1 = matrix[r][c];
							int f2 = matrix[s][c];
							// we can use row s, but we have to find the least common multiple
							for (int x = c; x < columns; x++) {
								newMatrix[rn][x] *= -f2;
								newMatrix[rn][x] += f1 * matrix[s][x];
							}
							done = true;
						}
					}

					// finally, try to add a weakening constraint
					for (int s = cs; s < cs + ts && !done; s++) {
						if (matrix[s][c] == -1) {
							int f1 = matrix[r][c];
							// we can use row s, but we have to find the least common multiple
							for (int x = c; x < columns; x++) {
								newMatrix[rn][x] += f1 * matrix[s][x];
							}
							done = true;

							// swap with lastStrong to avoid over-use of weak constraints
							int[] tmp = newMatrix[lastStrong];
							newMatrix[lastStrong] = newMatrix[rn];
							newMatrix[rn] = tmp;
							// move last strong pointer up
							lastStrong--;
							// decrease r, as the new row might have a non 0 value at [r][c]
							r--;
						}
					}

					// if all else fails, we have a positive value left on matrix[r][c] and no way to
					// reduce it, even by weakening. This implies a tau-transition, remove the row.
					if (!done) {
						// eliminate row r;
						Arrays.fill(newMatrix[rn], 0);

						if (lastStrong >= 0) {
							// swap with lastStrong to avoid over-use of weak constraints
							int[] tmp = newMatrix[lastStrong];
							newMatrix[lastStrong] = newMatrix[rn];
							newMatrix[rn] = tmp;
							// move last strong pointer up
							lastStrong--;
						}
						// decrease r, as the new row might have a non 0 value at [r][c]
						r--;
					}

				} else if (matrix[r][c] < 0) {

					if (colNames[c] == VISIBLE) {
						// mapped transition that has no corresponding event in the log.
						// set value of 0. 
						matrix[r][c] = 0;
						done = true;
					}

					// element at row r < 0
					// reduce by adding and element from row 0..cs-1
					for (int s = 0; s < cs && !done; s++) {
						if (first1[s] == c) {
							// add this row
							int f1 = matrix[r][c];
							for (int x = c; x < columns; x++) {
								newMatrix[rn][x] -= f1 * matrix[s][x];
							}
							done = true;
						}
					}

					// if not done, try to add another constraint
					for (int s = cs + ts; s < rows && !done; s++) {
						if (matrix[s][c] > 0) {
							int f1 = matrix[r][c];
							int f2 = matrix[s][c];
							// we can use row s, but we have to find the least common multiple
							for (int x = c; x < columns; x++) {
								newMatrix[rn][x] *= f2;
								newMatrix[rn][x] -= f1 * matrix[s][x];
							}
							done = true;
						}
					}

					// if not done, set to 0 as this is only weakens the constraint
					if (!done) {
						newMatrix[rn][c] = 0;
						if (lastStrong >= 0) {
							// swap with lastStrong to avoid over-use of weak constraints
							int[] tmp = newMatrix[lastStrong];
							newMatrix[lastStrong] = newMatrix[rn];
							newMatrix[rn] = tmp;
							// move last strong pointer up
							lastStrong--;
						}
						// decrease r, as the new row might have a non 0 value at [r][c]
						r--;
					}
				}
			} // for rows
				// copy derived back in.
			for (int r = cs + ts; r < rows; r++) {
				// copy original
				matrix[r] = newMatrix[r - cs - ts];
			}
		} // for columns

		//		System.out.println("After:");
		//		printMatrix(matrix);

		for (int c = 0; c < c2id.size(); c++) {
			label2input.put(c, new HashSet<Constraint>());
			label2output.put(c, new HashSet<Constraint>());
		}

		String[] classColNames = Arrays.copyOfRange(colNames, ts, ts + cs);
		// now translate the matrix to constraints per label
		for (int r = cs + ts; r < rows; r++) {
			Constraint constraint = new Constraint(c2id.size(), matrix[r][columns - 1], classColNames);
			for (int c = ts; c < ts + cs; c++) {
				if (matrix[r][c] > 0) {
					constraint.addInput((c - ts), matrix[r][c]);
				} else if (matrix[r][c] < 0) {
					constraint.addOutput((c - ts), -matrix[r][c]);
				}
			}
			if (constraints.add(constraint)) {
				// new constraint;
				for (int c = ts; c < ts + cs; c++) {
					if (matrix[r][c] > 0) {
						label2input.get((c - ts)).add(constraint);
					} else if (matrix[r][c] < 0) {
						label2output.get((c - ts)).add(constraint);
					}
				}
			}
		}

		//		System.out.println("Found " + constraints.size() + " constraints.");
		//		System.out.println(this.toString());

	}

	private void printMatrix(int[][] matrix) {

		for (int c = 0; c < colNames.length; c++) {
			System.out.print(colNames[c]);
			System.out.print(",");
		}
		System.out.println();
		for (int c = 0; c < colNames.length; c++) {
			System.out.print("" + c);
			System.out.print(",");
		}
		System.out.println();

		for (int r = 0; r < matrix.length; r++) {
			for (int c = 0; c < matrix[r].length; c++) {
				System.out.print(matrix[r][c]);
				System.out.print(",");
			}
			System.out.println();
		}
	}

	public void reset() {
		for (Constraint constraint : constraints) {
			constraint.reset();
		}
	}

	public boolean satisfiedAfterOccurence(int label) {
		// process all relevant constraints for internal state consistency
		boolean satisfied = true;
		for (Constraint constraint : label2input.get(label)) {
			if (constraint.satisfied()) {
				// if it is satisfied, make sure it remains so
				satisfied &= constraint.satisfiedAfterOccurence(label);
			} else {
				// if it is not satisfied, update the state
				constraint.satisfiedAfterOccurence(label);
			}
		}
		for (Constraint constraint : label2output.get(label)) {
			if (constraint.satisfied()) {
				// if it is satisfied, make sure it remains so
				satisfied &= constraint.satisfiedAfterOccurence(label);
			} else {
				// if it is not satisfied, update the state
				constraint.satisfiedAfterOccurence(label);
			}
		}
		return satisfied;
	}

	public int size() {
		return constraints.size();
	}

	public String toString() {
		return constraints.toString();
	}
}
