// Distributed under the MIT License.
// See LICENSE.txt for details.
/// \cond
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Options/String.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/DistributedObject.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// Forward declaration: promise that this way to access the global cache will
// be defined elsewhere. This was present in the spectre minimal executable.
namespace Parallel {
template <typename Metavariables>
class CProxy_GlobalCache;
}  // namespace Parallel

// Forward declarations: promise that the parallel components and actions will 
// eventually be defined. This lets you refer to them before you actually 
// define them.
template <typename Metavars>
struct PiEstimator;

template <typename Metavars>
struct DartThrower;

namespace Actions {
struct ThrowDarts;
struct ProcessHitsAndThrows;
} // Actions

////////////////////////////////////////////////////////////////////////
// TUTORIAL STEP 1: Set up quantities stored in DataBox
////////////////////////////////////////////////////////////////////////
namespace Tags {
// TUTORIAL: add structs for the four quantities we will need in memory:
// ThrowsAllProcs, HitsAllProcs, DartsPerIteration, and AccuracyGoal.
struct ThrowsAllProcs: db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<>;
  static constexpr bool pass_metavariables = false;
  static size_t create_from_options() { return 0; }
};

struct HitsAllProcs: db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<>;
  static constexpr bool pass_metavariables = false;
  static size_t create_from_options() { return 0; }
};

struct DartsPerIteration: db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<>;
  static constexpr bool pass_metavariables = false;
  static size_t create_from_options() { return 1000000; }
};

struct AccuracyGoal: db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options() { return 1.e-6; }
};
}  // Namespace Tags

////////////////////////////////////////////////////////////////////////
// TUTORIAL STEP 2: Set up Actions ("tasks")
////////////////////////////////////////////////////////////////////////
namespace Actions {

// In spectre, "iterable actions" (actions that can be done more than once) are 
// made by creating a struct with a function apply with the following
// template parameters (compile-time parameters), parameters, and 
// return type.
struct ThrowDarts {
  template <typename DbTags, typename... InboxTags, typename Metavars,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavars>& cache,
      const ArrayIndex& array_index, const ActionList& /*meta*/,
      const ParallelComponent* const /*meta*/
  ) {
    // TUTORIAL: get how many darts to throw from the DataBox
    const size_t number_of_darts = db::get<Tags::DartsPerIteration>(box);

    // TUTORIAL: paste code from PiDartSerial.hpp to actually throw N darts
    // at the unit square, seeing how many hit the quarter circle
    std::random_device device;
    std::mt19937_64 generator(device());
    std::uniform_real_distribution distribution{0.0, 1.0};
    size_t hits = 0;
    for (size_t i = 0; i < number_of_darts; ++i) {
      const double x = distribution(generator);
      const double y = distribution(generator);
      if (x * x + y * y < 1) {
        hits += 1;
      }
    }

    // TUTORIAL: Send the data to the reduction action ProcessHitsAndThrows
    const auto& pi_estimator_proxy =
        Parallel::get_parallel_component<PiEstimator<Metavars>, Metavars>(
            cache);
    const auto& dart_thrower_element_proxy =
        Parallel::get_parallel_component<DartThrower<Metavars>, Metavars>(
            cache)[array_index];
    Parallel::ReductionData<Parallel::ReductionDatum<size_t, funcl::Plus<>>>
        hits_to_send{hits};
    Parallel::contribute_to_reduction<Actions::ProcessHitsAndThrows>(
        hits_to_send, dart_thrower_element_proxy, pi_estimator_proxy);

    // After this action completes, tell this element of the
    // DartThrower array parallel component to pause until further notice.
    // (That notice might come frm the ProcessHitsAndThrows action.)
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

// In spectre, "reduction actions" (actions that receive data from the 
// elements of an array parallel component and then reduce them to a single
// result) are made by creating a struct with a function apply with the
// following template parameters (compile-time parameters), parameters, and 
// return type.
struct ProcessHitsAndThrows {
  template <typename ParallelComponent, typename DbTags, typename Metavars,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavars>& cache,
                    const ArrayIndex& /*array_index*/, const size_t new_hits) {
    // TUTORIAL: get number of processors from the cache
    const size_t num_procs = Parallel::number_of_procs<size_t>(cache);

    // TUTORIAL: get number of darts thrown each iteration from the DataBox
    const size_t darts_per_iteration = db::get<Tags::DartsPerIteration>(box);

    // TUTORIAL: complete this function that updates quantities in the DataBox
    db::mutate_apply<tmpl::list<Tags::HitsAllProcs, Tags::ThrowsAllProcs>,
                     tmpl::list<>>(
        [&new_hits, &darts_per_iteration, &num_procs](
            const gsl::not_null<size_t*> hits_all_procs,
            const gsl::not_null<size_t*> throws_all_procs) {
          *hits_all_procs += new_hits;
          *throws_all_procs += darts_per_iteration * num_procs;
        },
        make_not_null(&box));

    // TUTORIAL: estiamte pi, compute the fractional accuracy, and 
    // print the result using Parallel::printf
    const double pi_estimate =
        4.0 * static_cast<double>(db::get<Tags::HitsAllProcs>(box)) /
        static_cast<double>(db::get<Tags::ThrowsAllProcs>(box));
    const double fractional_accuracy = abs(pi_estimate - M_PI) / M_PI;

    Parallel::printf("Pi ~ %1.15f (accuracy: %1.15f)\n", pi_estimate,
                     fractional_accuracy);

    // TUTORIAL: if fractional accuracy is bigger than the accuracy goal,
    // tell each element of the DartThrower parallel component to unpause
    // (that is, throw some more darts).
    if (fractional_accuracy > db::get<Tags::AccuracyGoal>(box)) {
      for (size_t i = 0; i < num_procs; ++i) {
        Parallel::get_parallel_component<DartThrower<Metavars>>(cache)[i]
            .perform_algorithm(true);
      }
    }
  }
};
} // namespace Actions

////////////////////////////////////////////////////////////////////////
// TUTORIAL STEP 3: Set up parallel components
////////////////////////////////////////////////////////////////////////

// TUTORIAL: Create PiEstimator parallel component struct
template <typename Metavars>
struct PiEstimator {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavars;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Execute, tmpl::list<>>>;
  using simple_tags_from_options = 
      tmpl::list<Tags::HitsAllProcs, Tags::ThrowsAllProcs,       
                 Tags::DartsPerIteration, Tags::AccuracyGoal>;
  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavars>& global_cache);
};

// TUTORIAL: after uncommenting the above, add the code between /* and */
// below to the PiEstimator struct.
/*
*/

// TUTORIAL: After creating PiEstimator, uncomment this function by
// removing the /* and */ surrounding it.
//
// This function is necessary "boilerplate" that tells
// spectre when one phase ends, start the next one.
template <typename Metavars>
void PiEstimator<Metavars>::execute_next_phase(
    const Parallel::Phase next_phase,
    const Parallel::CProxy_GlobalCache<Metavars>& global_cache) {
  auto& local_cache = *Parallel::local_branch(global_cache);
  Parallel::get_parallel_component<PiEstimator<Metavars>>(local_cache)
      .start_phase(next_phase);
}

// TUTORIAL: Create DartThrower parallel component struct
template <typename Metavars>
struct DartThrower {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavars;
  using phase_dependent_action_list =     
      tmpl::list<Parallel::PhaseActions<Parallel::Phase::Execute,   
                 tmpl::list<Actions::ThrowDarts>>>;
  using simple_tags_from_options = tmpl::list<Tags::DartsPerIteration>;
  using array_index = int;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavars>& global_cache);
  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavars>& global_cache,
      const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
          initialization_options,
      const std::unordered_set<size_t>& procs_to_ignore = {});
};

// TUTORIAL: Add the code between /* and */ to the DartThrower struct.
/*
*/

// TUTORIAL: After creating DartThrower, uncomment this function by
// removing the /* and */ surrounding it. Then add it to the parallel
// component struct as a static function. 
// 
// This function is necessary "boilerplate" that tells
// spectre when one phase ends, start the next one.
template <typename Metavars>
void DartThrower<Metavars>::execute_next_phase(
    const Parallel::Phase next_phase,
    const Parallel::CProxy_GlobalCache<Metavars>& global_cache) {
  auto& local_cache = *Parallel::local_branch(global_cache);
  Parallel::get_parallel_component<DartThrower<Metavars>>(local_cache)
      .start_phase(next_phase);
}

// TUTORIAL: After creating DartThrower, uncomment this function by
// removing the /* and */ surrounding it. Then add it to the parallel
// component struct as a static function. 
//
// This function assigns the array elements to
// specific cores (processors). The strategy is "round robin:" assign
// one per core until each core (except any the user wants to skip) has one,
// then repeat until each has two, etc.
//
// Note: since we choose here that there will be one DartThrower element 
// per core, each core will get one element, unless the user asks to skip
// one or more cores.


template <typename Metavars>
void DartThrower<Metavars>::allocate_array(
    Parallel::CProxy_GlobalCache<Metavars>& global_cache,
    const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
        initialization_options,
    const std::unordered_set<size_t>& procs_to_ignore) {
  auto& local_cache = *Parallel::local_branch(global_cache);
  auto& array_proxy =
      Parallel::get_parallel_component<DartThrower<Metavars>>(local_cache);

  size_t which_proc = 0;
  const size_t num_procs = Parallel::number_of_procs<size_t>(local_cache);
  const size_t number_of_elements = num_procs;

  for (size_t i = 0; i < number_of_elements; ++i) {
    while (procs_to_ignore.find(which_proc) != procs_to_ignore.end()) {
      which_proc = which_proc + 1 == num_procs ? 0 : which_proc + 1;
    }
    array_proxy[i].insert(global_cache, initialization_options, which_proc);
    which_proc = which_proc + 1 == num_procs ? 0 : which_proc + 1;
  }
  array_proxy.doneInserting();
}

struct Metavariables {
  // TUTORIAL: ADD PARALLEL COMPONENTS TO THE LIST AFTER COMPLETING THEM.
  using component_list = 
      tmpl::list<PiEstimator<Metavariables>, DartThrower<Metavariables>>;

  static constexpr std::array<Parallel::Phase, 3> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Execute,
       Parallel::Phase::Exit}};

  static constexpr Options::String help{"Compute Pi with Monte Carlo"};

  // tell code metavariables have no run-time content that must be sent
  // over the network when remote objects want the metavariables. Do this
  // by defining a pup (pack-unpack) function that does nothing.
  void pup(PUP::er& /*p*/) {}
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling};
static const std::vector<void (*)()> charm_init_proc_funcs{};
