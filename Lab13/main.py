import arviz as az
from matplotlib import pyplot as plt

centered_eight_data = az.load_arviz_data("centered_eight")

non_centered_eight_data = az.load_arviz_data("non_centered_eight")

print("Informații despre modelul centrat:")
print("Număr de lanțuri:", centered_eight_data.posterior.chain.size)
print("Mărime totală a eșantionului generat:", centered_eight_data.posterior.draw.size)

az.plot_posterior(centered_eight_data)

print("\nInformații despre modelul necentrat:")
print("Număr de lanțuri:", non_centered_eight_data.posterior.chain.size)
print("Mărime totală a eșantionului generat:", non_centered_eight_data.posterior.draw.size)

az.plot_posterior(non_centered_eight_data)

plt.show()

print("Criteriul R-hat pentru modelul centrat:")
print(az.rhat(centered_eight_data, var_names=["mu", "tau"]))

print("\nCriteriul R-hat pentru modelul necentrat:")
print(az.rhat(non_centered_eight_data, var_names=["mu", "tau"]))

autocorr_mu_centered = az.autocorr(centered_eight_data.posterior["mu"].values)
autocorr_mu_non_centered = az.autocorr(non_centered_eight_data.posterior["mu"].values)

print("\nAutocorrelation for the 'mu' parameter in the centered model:")
print(autocorr_mu_centered)

print("\nAutocorrelation for the 'mu' parameter in the non-centered model:")
print(autocorr_mu_non_centered)

autocorr_tau_centered = az.autocorr(centered_eight_data.posterior["tau"].values)
autocorr_tau_non_centered = az.autocorr(non_centered_eight_data.posterior["tau"].values)

print("\nAutocorrelation for the 'tau' parameter in the centered model:")
print(autocorr_tau_centered)

print("\nAutocorrelation for the 'tau' parameter in the non-centered model:")
print(autocorr_tau_non_centered)

divergences_centered = centered_eight_data.sample_stats.diverging.sum()
divergences_non_centered = non_centered_eight_data.sample_stats.diverging.sum()

print(f"\nNumărul de divergențe pentru modelul centrat: {divergences_centered}")
print(f"Numărul de divergențe pentru modelul necentrat: {divergences_non_centered}")

az.plot_pair(centered_eight_data, var_names=["mu", "tau"], divergences=True, textsize=8)
plt.suptitle("Pair Plot for Centered Model")

az.plot_pair(non_centered_eight_data, var_names=["mu", "tau"], divergences=True, textsize=8)
plt.suptitle("Pair Plot for Non-Centered Model")

plt.show()
