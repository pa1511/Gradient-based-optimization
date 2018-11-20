package gradient.updater;

import javax.annotation.Nonnull;

public interface IGradientBasedUpdater {
	
	/**
	 * Updates the values array
	 */
	public void update(@Nonnull double[] values,@Nonnull double[] gradients,double fintess, double value);

}
