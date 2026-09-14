"""
korg_workflow.py

A Python-side workflow wrapper for using Korg.jl (stellar spectral synthesis)
through juliacall/PythonCall.

Handles the full chain that tends to trip people up when Korg is called
from a Python notebook instead of a native Julia REPL:

    1. Activating the correct Julia project environment
    2. Optionally nuking a stale Manifest.toml and re-instantiating
    3. Making sure Korg is actually a resolved dependency (adds it if not)
    4. Loading Korg with `using Korg`
    5. Running Korg.synth(...) with sane defaults
    6. A `print_repl_checklist()` helper documenting the equivalent manual
       steps to run directly in a Julia REPL if something needs debugging
       outside of juliacall (precompilation progress bars, in particular,
       are much easier to read there than through a notebook).

Typical usage
-------------
    from korg_workflow import KorgWorkflow

    wf = KorgWorkflow(project_path="/Volumes/SSD2/MyPython/Julia")
    wf.setup(remove_manifest=True)   # only needed once, or after a break
    wf.load()

    wls, flux, continuum = wf.synth(
        Teff=5000,
        logg=4.32,
        M_H=-1.1,
        C=-0.5,
        wavelengths=(5850, 5900),
    )

    import matplotlib.pyplot as plt
    plt.plot(wls, flux / continuum)
    plt.xlabel("Wavelength (A)")
    plt.ylabel("Normalized flux")
    plt.show()
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

logger = logging.getLogger("korg_workflow")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class KorgWorkflow:
    """Manage a juliacall session for Korg.jl spectral synthesis."""

    def __init__(self, project_path: str, linelist_source: str = "GALAH_DR3"):
        """
        Parameters
        ----------
        project_path : str
            Path to the Julia project environment containing (or intended
            to contain) Korg as a dependency, e.g. "/Volumes/SSD2/MyPython/Julia".
        linelist_source : str
            Which Korg linelist getter to use by default. Currently supports
            "GALAH_DR3" (Korg.get_GALAH_DR3_linelist). Extend as needed.
        """
        self.project_path = str(Path(project_path).expanduser())
        self.linelist_source = linelist_source

        self._jl = None          # juliacall.Main handle
        self._Korg = None        # Korg module handle, set after load()
        self._linelist = None    # cached linelist, loaded lazily

    # ------------------------------------------------------------------
    # Setup / environment management
    # ------------------------------------------------------------------

    def _get_jl(self):
        if self._jl is None:
            from juliacall import Main as jl
            self._jl = jl
        return self._jl

    def setup(self, remove_manifest: bool = False, verbose: bool = True) -> None:
        """
        Activate the Julia project and make sure dependencies are resolved.

        Parameters
        ----------
        remove_manifest : bool
            If True, deletes Manifest.toml before instantiating. Use this
            after switching Julia/package versions or if you hit
            'Package Korg not found in current path' despite Korg being
            listed in Project.toml — a stale Manifest is the usual cause.
        verbose : bool
            Print progress to the logger.
        """
        jl = self._get_jl()
        manifest_path = Path(self.project_path) / "Manifest.toml"

        if remove_manifest and manifest_path.exists():
            if verbose:
                logger.info(f"Removing stale Manifest.toml at {manifest_path}")
            manifest_path.unlink()

        if verbose:
            logger.info(f"Activating Julia project at {self.project_path}")
            logger.info(
                "Instantiating environment — first run can take several "
                "minutes (precompilation of Korg's dependency tree). This "
                "may look 'stuck' in a notebook; progress bars from Pkg "
                "don't always stream cleanly through juliacall."
            )

        jl.seval(f"""
        import Pkg
        Pkg.activate(raw"{self.project_path}")
        Pkg.instantiate()
        """)

        self._ensure_korg_available(verbose=verbose)

        if verbose:
            logger.info("Environment ready.")

    def _ensure_korg_available(self, verbose: bool = True) -> None:
        """Add Korg to the active project if it isn't already a dependency."""
        jl = self._get_jl()
        has_korg = jl.seval("""
        haskey(Pkg.project().dependencies, "Korg")
        """)
        if not bool(has_korg):
            if verbose:
                logger.info("Korg not found in project deps — adding it now.")
            jl.seval('Pkg.add("Korg")')

    # ------------------------------------------------------------------
    # Loading Korg
    # ------------------------------------------------------------------

    def load(self, verbose: bool = True) -> Any:
        """
        Run `using Korg` in the activated environment and cache the handle.

        Returns the Korg module handle (also stored as self.Korg).
        """
        jl = self._get_jl()
        if verbose:
            logger.info(
                "Loading Korg (using Korg) — first import in a session "
                "triggers JIT compilation on top of precompilation, so "
                "this call itself can be slow the first time."
            )
        jl.seval("using Korg")
        self._Korg = jl.Korg
        jl.seval("flush(stdout)")
        if verbose:
            logger.info("Korg loaded.")
        return self._Korg

    @property
    def Korg(self) -> Any:
        if self._Korg is None:
            raise RuntimeError(
                "Korg has not been loaded yet — call .load() first."
            )
        return self._Korg

    # ------------------------------------------------------------------
    # Linelist handling
    # ------------------------------------------------------------------

    def get_linelist(self, force_reload: bool = False) -> Any:
        """
        Fetch (and cache) the configured linelist. First call may trigger
        a data download and can be slow/silent over a poor connection.
        """
        if self._linelist is not None and not force_reload:
            return self._linelist

        if self.linelist_source == "GALAH_DR3":
            logger.info(
                "Fetching GALAH DR3 linelist (first call may download data)..."
            )
            self._linelist = self.Korg.get_GALAH_DR3_linelist()
        else:
            raise ValueError(
                f"Unknown linelist_source: {self.linelist_source!r}"
            )

        return self._linelist

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synth(
        self,
        Teff: float,
        logg: float,
        M_H: float = 0.0,
        C: Optional[float] = None,
        wavelengths: Sequence[float] = (5000, 5100),
        linelist: Optional[Any] = None,
        **extra_kwargs: Any,
    ) -> Tuple[Any, Any, Any]:
        """
        Thin wrapper around Korg.synth with GALAH-style defaults.

        Any additional keyword arguments (e.g. vmic, abundances) are passed
        straight through to Korg.synth.

        Returns
        -------
        (wls, flux, continuum)
        """
        if linelist is None:
            linelist = self.get_linelist()

        kwargs = dict(
            Teff=Teff,
            logg=logg,
            M_H=M_H,
            linelist=linelist,
            wavelengths=tuple(wavelengths),
        )
        if C is not None:
            kwargs["C"] = C
        kwargs.update(extra_kwargs)

        logger.info(
            f"Running Korg.synth(Teff={Teff}, logg={logg}, M_H={M_H}, "
            f"wavelengths={tuple(wavelengths)})"
        )
        wls, flux, continuum = self.Korg.synth(**kwargs)
        return wls, flux, continuum

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        wls: Any,
        flux: Any,
        continuum: Any,
        normalize: bool = True,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """
        Plot a synthesized spectrum from synth()'s output.

        Parameters
        ----------
        wls, flux, continuum : array-like
            Output of .synth(). Pass continuum=None to skip normalizing
            even if normalize=True.
        normalize : bool
            If True (default), plots flux / continuum. If False, plots
            raw flux.
        title : str, optional
            Plot title.
        save_path : str, optional
            If given, saves the figure to this path (e.g. "spectrum.png")
            in addition to/instead of showing it.
        show : bool
            Whether to call plt.show(). Set False for headless/batch runs
            where you only want save_path written to disk.

        Returns
        -------
        (fig, ax) matplotlib objects, for further customization.
        """
        import matplotlib.pyplot as plt

        y = (flux / continuum) if (normalize and continuum is not None) else flux
        ylabel = "Normalized flux" if (normalize and continuum is not None) else "Flux"

        fig, ax = plt.subplots()
        ax.plot(wls, y)
        ax.set_xlabel("Wavelength (\u00c5)")
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved plot to {save_path}")

        if show:
            plt.show()

        return fig, ax

    def synth_and_plot(
        self,
        Teff: float,
        logg: float,
        M_H: float = 0.0,
        C: Optional[float] = None,
        wavelengths: Sequence[float] = (5000, 5100),
        linelist: Optional[Any] = None,
        normalize: bool = True,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        **extra_kwargs: Any,
    ) -> Tuple[Any, Any, Any]:
        """
        Convenience wrapper: runs synth() then immediately plots the result.

        Returns (wls, flux, continuum), same as synth().
        """
        wls, flux, continuum = self.synth(
            Teff=Teff,
            logg=logg,
            M_H=M_H,
            C=C,
            wavelengths=wavelengths,
            linelist=linelist,
            **extra_kwargs,
        )
        self.plot(
            wls, flux, continuum,
            normalize=normalize,
            title=title or f"Teff={Teff}, logg={logg}, [M/H]={M_H}",
            save_path=save_path,
            show=show,
        )
        return wls, flux, continuum

    # ------------------------------------------------------------------
    # Manual REPL fallback instructions
    # ------------------------------------------------------------------

    def print_repl_checklist(self) -> None:
        """
        Print the equivalent manual steps to run in a native Julia REPL.

        Useful when something needs debugging outside juliacall — Pkg's
        precompilation progress bars and error messages are much easier
        to read in a real REPL than through a notebook.
        """
        checklist = f"""
        Manual Julia REPL checklist
        ============================
        1. Start `julia` in a terminal (not through PythonCall).

        2. Activate the environment and clear a stale manifest if needed:

               import Pkg
               Pkg.activate("{self.project_path}")
               rm("{self.project_path}/Manifest.toml", force=true)
               Pkg.instantiate()

        3. Confirm Korg is a project dependency, adding it if not:

               Pkg.status()          # look for Korg in the list
               Pkg.add("Korg")       # only if it's missing

        4. Verify Korg loads and synth is callable:

               using Korg
               Korg.synth

        5. Once this succeeds natively, the same environment path can be
           reused from Python via KorgWorkflow(project_path=...).setup().
        """
        print(checklist)


# ----------------------------------------------------------------------
# Demo / CLI entry point
# ----------------------------------------------------------------------

def _demo() -> None:
    wf = KorgWorkflow(project_path="/Volumes/SSD2/MyPython/Julia")
    wf.print_repl_checklist()
    wf.setup(remove_manifest=False)
    wf.load()

    wls, flux, continuum = wf.synth_and_plot(
        Teff=5000,
        logg=4.32,
        M_H=-1.1,
        C=-0.5,
        wavelengths=(5850, 5900),
    )

    print(len(wls), len(flux), len(continuum))
    print(wls[:5])


if __name__ == "__main__":
    _demo()
