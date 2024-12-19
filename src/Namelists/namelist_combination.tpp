// -*-c++-*-
#include "copyright.h"
#include "namelist_combination.h"

namespace stormm {
namespace namelist {

//-------------------------------------------------------------------------------------------------
template <typename T>
double arbitrateCutoff(const T &foocon, const PPPMControls &pmecon, const NonbondedTheme theme) {
  double result;
  const NamelistEmulator foo_nml = foocon.getTranscript();
  const NamelistEmulator pme_nml = pmecon.getTranscript();
  if (pmecon.getTheme() != theme ||
      pme_nml.getKeywordStatus("cut") != InputStatus::USER_SPECIFIED ) {
    
    // Determine whether the more general control block specifies the cutoff.  If so, take it.
    switch (theme) {
    case NonbondedTheme::ELECTROSTATIC:
      if (foocon.getKeywordStatus("cut") == InputStatus::USER_SPECIFIED ||
          foocon.getKeywordStatus("elec_cut") == InputStatus::USER_SPECIFIED) {
        return foocon.getElectrostaticCutoff();
      }
      break;
    case NonbondedTheme::VAN_DER_WAALS:
      if (foocon.getKeywordStatus("cut") == InputStatus::USER_SPECIFIED ||
          foocon.getKeywordStatus("vdw_cut") == InputStatus::USER_SPECIFIED) {
        return foocon.getVanDerWaalsCutoff();
      }
      break;
    case NonbondedTheme::ALL:
      problem = true;
      break;
    }
  }
  else if (pmecon.getTheme() != theme) {
    switch (theme) {
    case NonbondedTheme::ELECTROSTATIC:
      return foocon.getElectrostaticCutoff();
    case NonbondedTheme::VAN_DER_WAALS:
      return foocon.getVanDerWaalsCutoff();
    case NonbondedTheme::ALL:
      problem = true;
      break;
    }
  }

  // Report errors in the input, which is more about how the developer set up the function call.
  if (problem) {
    rErr("Specify whether to seek the non-bonded cutoff for electrostatic or van-der Waals "
         "interactions.  But are unique quantities, though assigning them to the same value or "
         "to distinct values may change the way in which non-bonded interactions are processed "
         "accumulated.", "arbitrateCutoff");
  }
  
  // The cutoff is specified in the PPPM control block and it is relevant, or the more general
  // control block did not contain a user-specified cutoff of the relevant type.  Take the value
  // from the PPPM control block.
  return pmecon.getCutoff();
}

//-------------------------------------------------------------------------------------------------
template <typename T>
double arbitrateCutoff(const T &foocon, const PPPMControls &pmecon_a, const PPPMControls &pmecon_b,
                       const NonbondedTheme theme) {
  if (pmecon_a.getTheme() == theme) {
    return arbitrateCutoff(foocon, pmecon_a, theme);
  }
  else {
    return arbitrateCutoff(foocon, pmecon_b, theme);
  }
  __builtin_unreachable();
}

} // namespace namelist
} // namespace stormm
