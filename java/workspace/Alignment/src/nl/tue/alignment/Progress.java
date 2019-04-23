package nl.tue.alignment;

public interface Progress extends Canceler {

	public final static Progress INVISIBLE = new Progress() {

		public void setMaximum(int maximum) {
		}

		public void inc() {
		}

		public boolean isCanceled() {
			return false;
		}

		public void log(String message) {
			System.out.println(message);
		}

	};

	public void setMaximum(int maximum);

	public void inc();

	public void log(String message);
}
