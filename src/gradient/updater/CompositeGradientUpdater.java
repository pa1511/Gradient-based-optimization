package gradient.updater;

import java.util.Collection;

import javax.annotation.Nonnull;

import utilities.PArrays;
import utilities.random.RNGProvider;


public class CompositeGradientUpdater implements IGradientBasedUpdater{

	
	private final @Nonnull IGradientBasedUpdater[] updaters;
	
	public CompositeGradientUpdater(@Nonnull Collection<IGradientBasedUpdater> updaters) {
		this.updaters = PArrays.toArray(updaters,IGradientBasedUpdater.class);
	}

	@Override
	public void update(double[] values, double[] gradients, double fintess, double value) {
		updaters[RNGProvider.getRandom().nextInt(updaters.length)].update(values, gradients, fintess, value);
	}
	
}
