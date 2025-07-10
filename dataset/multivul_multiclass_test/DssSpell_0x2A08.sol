// SPDX-License-Identifier: AGPL-3.0-or-later
pragma solidity =0.8.16 >=0.5.12 ^0.8.16;

// lib/dss-exec-lib/src/CollateralOpts.sol

//
// CollateralOpts.sol -- Data structure for onboarding collateral
//
// Copyright (C) 2020-2022 Dai Foundation
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

struct CollateralOpts {
    bytes32 ilk;
    address gem;
    address join;
    address clip;
    address calc;
    address pip;
    bool    isLiquidatable;
    bool    isOSM;
    bool    whitelistOSM;
    uint256 ilkDebtCeiling;
    uint256 minVaultAmount;
    uint256 maxLiquidationAmount;
    uint256 liquidationPenalty;
    uint256 ilkStabilityFee;
    uint256 startingPriceFactor;
    uint256 breakerTolerance;
    uint256 auctionDuration;
    uint256 permittedDrop;
    uint256 liquidationRatio;
    uint256 kprFlatReward;
    uint256 kprPctReward;
}

// lib/dss-exec-lib/src/DssExec.sol

//
// DssExec.sol -- MakerDAO Executive Spell Template
//
// Copyright (C) 2020-2022 Dai Foundation
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

interface PauseAbstract {
    function delay() external view returns (uint256);
    function plot(address, bytes32, bytes calldata, uint256) external;
    function exec(address, bytes32, bytes calldata, uint256) external returns (bytes memory);
}

interface Changelog {
    function getAddress(bytes32) external view returns (address);
}

interface SpellAction {
    function officeHours() external view returns (bool);
    function description() external view returns (string memory);
    function nextCastTime(uint256) external view returns (uint256);
}

contract DssExec {

    Changelog      constant public log   = Changelog(0xdA0Ab1e0017DEbCd72Be8599041a2aa3bA7e740F);
    uint256                 public eta;
    bytes                   public sig;
    bool                    public done;
    bytes32       immutable public tag;
    address       immutable public action;
    uint256       immutable public expiration;
    PauseAbstract immutable public pause;

    // Provides a descriptive tag for bot consumption
    // This should be modified weekly to provide a summary of the actions
    // Hash: seth keccak -- "$(wget https://<executive-vote-canonical-post> -q -O - 2>/dev/null)"
    function description() external view returns (string memory) {
        return SpellAction(action).description();
    }

    function officeHours() external view returns (bool) {
        return SpellAction(action).officeHours();
    }

    function nextCastTime() external view returns (uint256 castTime) {
        return SpellAction(action).nextCastTime(eta);
    }

    // @param _description  A string description of the spell
    // @param _expiration   The timestamp this spell will expire. (Ex. block.timestamp + 30 days)
    // @param _spellAction  The address of the spell action
    constructor(uint256 _expiration, address _spellAction) {
        pause       = PauseAbstract(log.getAddress("MCD_PAUSE"));
        expiration  = _expiration;
        action      = _spellAction;

        sig = abi.encodeWithSignature("execute()");
        bytes32 _tag;                    // Required for assembly access
        address _action = _spellAction;  // Required for assembly access
        assembly { _tag := extcodehash(_action) }
        tag = _tag;
    }

    function schedule() public {
        require(block.timestamp <= expiration, "This contract has expired");
        require(eta == 0, "This spell has already been scheduled");
        eta = block.timestamp + PauseAbstract(pause).delay();
        pause.plot(action, tag, sig, eta);
    }

    function cast() public {
        require(!done, "spell-already-cast");
        done = true;
        pause.exec(action, tag, sig, eta);
    }
}

// lib/dss-test/lib/dss-interfaces/src/ERC/GemAbstract.sol

// A base ERC-20 abstract class
// https://eips.ethereum.org/EIPS/eip-20
interface GemAbstract {
    function totalSupply() external view returns (uint256);
    function balanceOf(address) external view returns (uint256);
    function allowance(address, address) external view returns (uint256);
    function approve(address, uint256) external returns (bool);
    function transfer(address, uint256) external returns (bool);
    function transferFrom(address, address, uint256) external returns (bool);
    function name() external view returns (string memory);
    function symbol() external view returns (string memory);
    function decimals() external view returns (uint8);
}

// lib/dss-test/lib/dss-interfaces/src/dss/OsmAbstract.sol

// https://github.com/makerdao/osm
interface OsmAbstract {
    function wards(address) external view returns (uint256);
    function rely(address) external;
    function deny(address) external;
    function stopped() external view returns (uint256);
    function src() external view returns (address);
    function hop() external view returns (uint16);
    function zzz() external view returns (uint64);
    function bud(address) external view returns (uint256);
    function stop() external;
    function start() external;
    function change(address) external;
    function step(uint16) external;
    function void() external;
    function pass() external view returns (bool);
    function poke() external;
    function peek() external view returns (bytes32, bool);
    function peep() external view returns (bytes32, bool);
    function read() external view returns (bytes32);
    function kiss(address) external;
    function diss(address) external;
    function kiss(address[] calldata) external;
    function diss(address[] calldata) external;
}

// lib/dss-exec-lib/src/DssExecLib.sol

//
// DssExecLib.sol -- MakerDAO Executive Spellcrafting Library
//
// Copyright (C) 2020-2022 Dai Foundation
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

interface Initializable {
    function init(bytes32) external;
}

interface Authorizable {
    function rely(address) external;
    function deny(address) external;
    function setAuthority(address) external;
}

interface Fileable {
    function file(bytes32, address) external;
    function file(bytes32, uint256) external;
    function file(bytes32, bytes32, uint256) external;
    function file(bytes32, bytes32, address) external;
}

interface Drippable {
    function drip() external returns (uint256);
    function drip(bytes32) external returns (uint256);
}

interface Pricing {
    function poke(bytes32) external;
}

interface ERC20 {
    function decimals() external returns (uint8);
}

interface DssVat {
    function hope(address) external;
    function nope(address) external;
    function ilks(bytes32) external returns (uint256 Art, uint256 rate, uint256 spot, uint256 line, uint256 dust);
    function Line() external view returns (uint256);
    function suck(address, address, uint256) external;
}

interface ClipLike {
    function vat() external returns (address);
    function dog() external returns (address);
    function spotter() external view returns (address);
    function calc() external view returns (address);
    function ilk() external returns (bytes32);
}

interface DogLike {
    function ilks(bytes32) external returns (address clip, uint256 chop, uint256 hole, uint256 dirt);
}

interface JoinLike {
    function vat() external returns (address);
    function ilk() external returns (bytes32);
    function gem() external returns (address);
    function dec() external returns (uint256);
    function join(address, uint256) external;
    function exit(address, uint256) external;
}

// Includes Median and OSM functions
interface OracleLike_0 {
    function src() external view returns (address);
    function lift(address[] calldata) external;
    function drop(address[] calldata) external;
    function setBar(uint256) external;
    function kiss(address) external;
    function diss(address) external;
    function kiss(address[] calldata) external;
    function diss(address[] calldata) external;
    function orb0() external view returns (address);
    function orb1() external view returns (address);
}

interface MomLike {
    function setOsm(bytes32, address) external;
    function setPriceTolerance(address, uint256) external;
}

interface RegistryLike {
    function add(address) external;
    function xlip(bytes32) external view returns (address);
}

// https://github.com/makerdao/dss-chain-log
interface ChainlogLike {
    function setVersion(string calldata) external;
    function setIPFS(string calldata) external;
    function setSha256sum(string calldata) external;
    function getAddress(bytes32) external view returns (address);
    function setAddress(bytes32, address) external;
    function removeAddress(bytes32) external;
}

interface IAMLike {
    function ilks(bytes32) external view returns (uint256,uint256,uint48,uint48,uint48);
    function setIlk(bytes32,uint256,uint256,uint256) external;
    function remIlk(bytes32) external;
    function exec(bytes32) external returns (uint256);
}

interface LerpFactoryLike {
    function newLerp(bytes32 name_, address target_, bytes32 what_, uint256 startTime_, uint256 start_, uint256 end_, uint256 duration_) external returns (address);
    function newIlkLerp(bytes32 name_, address target_, bytes32 ilk_, bytes32 what_, uint256 startTime_, uint256 start_, uint256 end_, uint256 duration_) external returns (address);
}

interface LerpLike {
    function tick() external returns (uint256);
}

interface RwaOracleLike {
    function bump(bytes32 ilk, uint256 val) external;
}

library DssExecLib {

    /* WARNING

The following library code acts as an interface to the actual DssExecLib
library, which can be found in its own deployed contract. Only trust the actual
library's implementation.

    */

    address constant public LOG = 0xdA0Ab1e0017DEbCd72Be8599041a2aa3bA7e740F;
    uint256 constant internal WAD      = 10 ** 18;
    uint256 constant internal RAY      = 10 ** 27;
    uint256 constant internal RAD      = 10 ** 45;
    uint256 constant internal THOUSAND = 10 ** 3;
    uint256 constant internal MILLION  = 10 ** 6;
    uint256 constant internal BPS_ONE_PCT             = 100;
    uint256 constant internal BPS_ONE_HUNDRED_PCT     = 100 * BPS_ONE_PCT;
    uint256 constant internal RATES_ONE_HUNDRED_PCT   = 1000000021979553151239153027;
    function dai()        public view returns (address) { return getChangelogAddress("MCD_DAI"); }
    function mkr()        public view returns (address) { return getChangelogAddress("MCD_GOV"); }
    function vat()        public view returns (address) { return getChangelogAddress("MCD_VAT"); }
    function jug()        public view returns (address) { return getChangelogAddress("MCD_JUG"); }
    function pot()        public view returns (address) { return getChangelogAddress("MCD_POT"); }
    function vow()        public view returns (address) { return getChangelogAddress("MCD_VOW"); }
    function end()        public view returns (address) { return getChangelogAddress("MCD_END"); }
    function reg()        public view returns (address) { return getChangelogAddress("ILK_REGISTRY"); }
    function daiJoin()    public view returns (address) { return getChangelogAddress("MCD_JOIN_DAI"); }
    function lerpFab()    public view returns (address) { return getChangelogAddress("LERP_FAB"); }
    function clip(bytes32 _ilk) public view returns (address _clip) {}
    function flip(bytes32 _ilk) public view returns (address _flip) {}
    function calc(bytes32 _ilk) public view returns (address _calc) {}
    function getChangelogAddress(bytes32 _key) public view returns (address) {}
    function setAuthority(address _base, address _authority) public {}
    function canCast(uint40 _ts, bool _officeHours) public pure returns (bool) {}
    function nextCastTime(uint40 _eta, uint40 _ts, bool _officeHours) public pure returns (uint256 castTime) {}
    function setValue(address _base, bytes32 _what, uint256 _amt) public {}
    function setValue(address _base, bytes32 _ilk, bytes32 _what, uint256 _amt) public {}
    function setDSR(uint256 _rate, bool _doDrip) public {}
    function setIlkStabilityFee(bytes32 _ilk, uint256 _rate, bool _doDrip) public {}
    function sendPaymentFromSurplusBuffer(address _target, uint256 _amount) public {}
    function linearInterpolation(bytes32 _name, address _target, bytes32 _ilk, bytes32 _what, uint256 _startTime, uint256 _start, uint256 _end, uint256 _duration) public returns (address) {}
}

// lib/dss-exec-lib/src/DssAction.sol

//
// DssAction.sol -- DSS Executive Spell Actions
//
// Copyright (C) 2020-2022 Dai Foundation
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

interface OracleLike_1 {
    function src() external view returns (address);
}

abstract contract DssAction {

    using DssExecLib for *;

    // Modifier used to limit execution time when office hours is enabled
    modifier limited {
        require(DssExecLib.canCast(uint40(block.timestamp), officeHours()), "Outside office hours");
        _;
    }

    // Office Hours defaults to true by default.
    //   To disable office hours, override this function and
    //    return false in the inherited action.
    function officeHours() public view virtual returns (bool) {
        return true;
    }

    // DssExec calls execute. We limit this function subject to officeHours modifier.
    function execute() external limited {
        actions();
    }

    // DssAction developer must override `actions()` and place all actions to be called inside.
    //   The DssExec function will call this subject to the officeHours limiter
    //   By keeping this function public we allow simulations of `execute()` on the actions outside of the cast time.
    function actions() public virtual;

    // Provides a descriptive tag for bot consumption
    // This should be modified weekly to provide a summary of the actions
    // Hash: seth keccak -- "$(wget https://<executive-vote-canonical-post> -q -O - 2>/dev/null)"
    function description() external view virtual returns (string memory);

    // Returns the next available cast time
    function nextCastTime(uint256 eta) external view returns (uint256 castTime) {
        require(eta <= type(uint40).max);
        castTime = DssExecLib.nextCastTime(uint40(eta), uint40(block.timestamp), officeHours());
    }
}

// src/DssSpell.sol
// SPDX-FileCopyrightText: © 2020 Dai Foundation <www.daifoundation.org>

//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

interface SUsdsLike {
    function drip() external returns (uint256);
    function file(bytes32 what, uint256 data) external;
}

interface DaiUsdsLike {
    function daiToUsds(address usr, uint256 wad) external;
}

interface MkrSkyLike {
    function mkrToSky(address usr, uint256 wad) external;
    function rate() external view returns (uint256);
}

interface StakingRewardsLike {
    function setRewardsDuration(uint256) external;
}

interface ProxyLike {
    function exec(address target, bytes calldata args) external payable returns (bytes memory out);
}

contract DssSpellAction is DssAction {
    // Provides a descriptive tag for bot consumption
    // This should be modified weekly to provide a summary of the actions
    // Hash: cast keccak -- "$(wget 'https://raw.githubusercontent.com/makerdao/community/93ebde6f0f265d990be3a3ec6524845ddc06e593/governance/votes/Executive%20vote%20-%20March%2020%2C%202025.md' -q -O - 2>/dev/null)"
    string public constant override description = "2025-03-20 MakerDAO Executive Spell | Hash: 0x222fd7fe53603675db01f1cc524770291cc40ba246dc4ffc350f955ee6217c84";

    // Set office hours according to the summary
    function officeHours() public pure override returns (bool) {
        return true;
    }

    // ---------- Rates ----------
    // Many of the settings that change weekly rely on the rate accumulator
    // described at https://docs.makerdao.com/smart-contract-modules/rates-module
    // To check this yourself, use the following rate calculation (example 8%):
    //
    // $ bc -l <<< 'scale=27; e( l(1.08)/(60 * 60 * 24 * 365) )'
    //
    // A table of rates can be found at
    //    https://ipfs.io/ipfs/QmVp4mhhbwWGTfbh2BzwQB9eiBrQBKiqcPRZCaAxNUaar6
    //
    // uint256 internal constant X_PCT_RATE = ;
    uint256 internal constant TWO_PT_SIX_TWO_PCT_RATE     = 1000000000820099554044024241;
    uint256 internal constant THREE_PT_FIVE_PCT_RATE      = 1000000001090862085746321732;
    uint256 internal constant FOUR_PT_FIVE_PCT_RATE       = 1000000001395766281313196627;
    uint256 internal constant FIVE_PT_SEVEN_FIVE_PCT_RATE = 1000000001772819380639683201;
    uint256 internal constant SIX_PCT_RATE                = 1000000001847694957439350562;
    uint256 internal constant SIX_PT_FIVE_PCT_RATE        = 1000000001996917783620820123;
    uint256 internal constant SIX_PT_SEVEN_FIVE_PCT_RATE  = 1000000002071266685321207000;
    uint256 internal constant SEVEN_PCT_RATE              = 1000000002145441671308778766;
    uint256 internal constant TEN_PT_SEVEN_FIVE_PCT_RATE  = 1000000003237735385034516037;
    uint256 internal constant ELEVEN_PCT_RATE             = 1000000003309234382829738808;
    uint256 internal constant ELEVEN_PT_FIVE_PCT_RATE     = 1000000003451750542235895695;

    // ---------- Math ----------
    uint256 internal constant WAD = 10 ** 18;

    // ---------- Contracts ----------
    GemAbstract internal immutable DAI                = GemAbstract(DssExecLib.dai());
    GemAbstract internal immutable MKR                = GemAbstract(DssExecLib.mkr());
    GemAbstract internal immutable SKY                = GemAbstract(DssExecLib.getChangelogAddress("SKY"));
    address internal immutable PIP_ETH                = DssExecLib.getChangelogAddress("PIP_ETH");
    address internal immutable PIP_WSTETH             = DssExecLib.getChangelogAddress("PIP_WSTETH");
    address internal immutable SUSDS                  = DssExecLib.getChangelogAddress("SUSDS");
    address internal immutable MCD_SPLIT              = DssExecLib.getChangelogAddress("MCD_SPLIT");
    address internal immutable REWARDS_LSMKR_USDS     = DssExecLib.getChangelogAddress("REWARDS_LSMKR_USDS");
    address internal immutable DAI_USDS               = DssExecLib.getChangelogAddress("DAI_USDS");
    address internal immutable MKR_SKY                = DssExecLib.getChangelogAddress("MKR_SKY");

    // ---------- Wallets ----------
    address internal constant BLUE                            = 0xb6C09680D822F162449cdFB8248a7D3FC26Ec9Bf;
    address internal constant BONAPUBLICA                     = 0x167c1a762B08D7e78dbF8f24e5C3f1Ab415021D3;
    address internal constant BYTERON                         = 0xc2982e72D060cab2387Dba96b846acb8c96EfF66;
    address internal constant CLOAKY_2                        = 0x9244F47D70587Fa2329B89B6f503022b63Ad54A5;
    address internal constant CLOAKY_KOHLA_2                  = 0x73dFC091Ad77c03F2809204fCF03C0b9dccf8c7a;
    address internal constant CLOAKY_ENNOIA                   = 0xA7364a1738D0bB7D1911318Ca3FB3779A8A58D7b;
    address internal constant INTEGRATION_BOOST_INITIATIVE    = 0xD6891d1DFFDA6B0B1aF3524018a1eE2E608785F7;
    address internal constant IMMUNEFI_COMISSION              = 0x7119f398b6C06095c6E8964C1f58e7C1BAa79E18;
    address internal constant IMMUNEFI_USER_PAYOUT_2025_03_20 = 0x29d17B5AcB1C68C574712B11F36C859F6FbdBe32;
    address internal constant JULIACHANG                      = 0x252abAEe2F4f4b8D39E5F12b163eDFb7fac7AED7;
    address internal constant WBC                             = 0xeBcE83e491947aDB1396Ee7E55d3c81414fB0D47;
    address internal constant PBG                             = 0x8D4df847dB7FfE0B46AF084fE031F7691C6478c2;

    // ---------- Constant Values ----------
    uint256 internal immutable MKR_SKY_RATE = MkrSkyLike(DssExecLib.getChangelogAddress("MKR_SKY")).rate();

    // ---------- Spark Proxy Spell ----------
    // Spark Proxy: https://github.com/marsfoundation/sparklend-deployments/blob/bba4c57d54deb6a14490b897c12a949aa035a99b/script/output/1/primary-sce-latest.json#L2
    address internal constant SPARK_PROXY = 0x3300f198988e4C9C63F75dF86De36421f06af8c4;
    address internal constant SPARK_SPELL = 0x1e865856d8F97FB34FBb0EDbF63f53E29a676aB6;

    function actions() public override {
        // ---------- ETH and WSTETH Oracle Migration ----------
        // Forum: https://forum.sky.money/t/march-20-2025-final-native-vault-engine-oracle-migration-proposal-eth-steth/26110?u=votewizard
        // Forum: https://forum.sky.money/t/technical-scope-of-the-eth-and-wsteth-oracles-migration/26128
        // Forum: https://forum.sky.money/t/march-20-2025-final-native-vault-engine-oracle-migration-proposal-eth-steth/26110/2
        // Poll: https://vote.makerdao.com/polling/QmV4uuru

        // Change ETH OSM source to 0x46ef0071b1E2fF6B42d36e5A177EA43Ae5917f4E
        OsmAbstract(PIP_ETH).change(0x46ef0071b1E2fF6B42d36e5A177EA43Ae5917f4E);

        // Change WSTETH OSM source to 0xA770582353b573CbfdCC948751750EeB3Ccf23CF
        OsmAbstract(PIP_WSTETH).change(0xA770582353b573CbfdCC948751750EeB3Ccf23CF);

        // ---------- Rates Changes  ----------
        // Forum: https://forum.sky.money/t/mar-20-2025-stability-scope-parameter-changes-24/26129
        // Forum: https://forum.sky.money/t/mar-20-2025-stability-scope-parameter-changes-24/26129/2

        // Reduce ETH-A Stability Fee by 1.75 percentage points from 7.75% to 6.00%
        DssExecLib.setIlkStabilityFee("ETH-A", SIX_PCT_RATE, /* doDrip = */ true);

        // Reduce ETH-B Stability Fee by 1.75 percentage points from 8.25% to 6.50%
        DssExecLib.setIlkStabilityFee("ETH-B", SIX_PT_FIVE_PCT_RATE, /* doDrip = */ true);

        // Reduce ETH-C Stability Fee by 1.75 percentage points from 7.50% to 5.75%
        DssExecLib.setIlkStabilityFee("ETH-C", FIVE_PT_SEVEN_FIVE_PCT_RATE, /* doDrip = */ true);

        // Reduce WSTETH-A Stability Fee by 1.75 percentage points from 8.75% to 7.00%
        DssExecLib.setIlkStabilityFee("WSTETH-A", SEVEN_PCT_RATE, /* doDrip = */ true);

        // Reduce WSTETH-B Stability Fee by 1.75 percentage points from 8.50% to 6.75%
        DssExecLib.setIlkStabilityFee("WSTETH-B", SIX_PT_SEVEN_FIVE_PCT_RATE, /* doDrip = */ true);

        // Reduce WBTC-A Stability Fee by 1.75 percentage points from 12.75% to 11.00%
        DssExecLib.setIlkStabilityFee("WBTC-A", ELEVEN_PCT_RATE, /* doDrip = */ true);

        // Reduce WBTC-B Stability Fee by 1.75 percentage points from 13.25% to 11.50%
        DssExecLib.setIlkStabilityFee("WBTC-B", ELEVEN_PT_FIVE_PCT_RATE, /* doDrip = */ true);

        // Reduce WBTC-C Stability Fee by 1.75 percentage points from 12.50% to 10.75%
        DssExecLib.setIlkStabilityFee("WBTC-C", TEN_PT_SEVEN_FIVE_PCT_RATE, /* doDrip = */ true);

        // Reduce ALLOCATOR-SPARK-A Stability Fee by 1.12 percentage points from 3.74% to 2.62%
        DssExecLib.setIlkStabilityFee("ALLOCATOR-SPARK-A", TWO_PT_SIX_TWO_PCT_RATE, /* doDrip = */ true);

        // Reduce DSR by 2.00 percentage points from 5.50% to 3.50%
        DssExecLib.setDSR(THREE_PT_FIVE_PCT_RATE, /* doDrip = */ true);

        // Reduce SSR by 2.00 percentage points from 6.50% to 4.50%
        SUsdsLike(SUSDS).drip();
        SUsdsLike(SUSDS).file("ssr", FOUR_PT_FIVE_PCT_RATE);

        // ---------- Smart Burn Engine Parameter Update ----------
        // Forum: https://forum.sky.money/t/smart-burn-engine-parameter-update-march-20-spell/26130
        // Forum: https://forum.sky.money/t/smart-burn-engine-parameter-update-march-20-spell/26130/3
        // Poll: https://vote.makerdao.com/polling/QmVrRf9L

        // Splitter: decrease hop for 432 seconds, from 2,160 seconds to 1,728 seconds
        DssExecLib.setValue(MCD_SPLIT, "hop", 1728);

        // Note: Update farm rewards duration
        StakingRewardsLike(REWARDS_LSMKR_USDS).setRewardsDuration(1728);

        // ---------- Bug Bounty Payout ----------
        // Forum: https://forum.sky.money/t/bounty-payout-request-for-immunefi-bug-38567/26072
        // Atlas: https://sky-atlas.powerhouse.io/A.2.9.1.1_Bug_Bounty_Program_For_Critical_Infrastructure/7d58645d-713c-4c54-a2ee-e0c948fb0c25%7C9e1f4492c8ce

        // Transfer 50,000 USDS to bug reporter at 0x29d17B5AcB1C68C574712B11F36C859F6FbdBe32
        _transferUsds(IMMUNEFI_USER_PAYOUT_2025_03_20, 50_000 * WAD);

        // Transfer 5,000 USDS to Immunefi at 0x7119f398b6C06095c6E8964C1f58e7C1BAa79E18
        _transferUsds(IMMUNEFI_COMISSION, 5_000 * WAD);

        // ---------- AD February 2025 Compensation ----------
        // Forum: https://forum.sky.money/t/february-2025-aligned-delegate-compensation/26131
        // Atlas: https://sky-atlas.powerhouse.io/A.1.5.8_Budget_For_Prime_Delegate_Slots/e3e420fc-9b1f-4fdc-9983-fcebc45dd3aa%7C0db3af4ece0c

        // BLUE - 4,000 USDS - 0xb6C09680D822F162449cdFB8248a7D3FC26Ec9Bf
        _transferUsds(BLUE, 4_000 * WAD);

        // Bonapublica - 4,000 USDS - 0x167c1a762B08D7e78dbF8f24e5C3f1Ab415021D3
        _transferUsds(BONAPUBLICA, 4_000 * WAD);

        // Cloaky - 4,000 USDS - 0x9244F47D70587Fa2329B89B6f503022b63Ad54A5
        _transferUsds(CLOAKY_2, 4_000 * WAD);

        // JuliaChang - 4,000 USDS - 0x252abAEe2F4f4b8D39E5F12b163eDFb7fac7AED7
        _transferUsds(JULIACHANG, 4_000 * WAD);

        // WBC - 3,613 USDS - 0xeBcE83e491947aDB1396Ee7E55d3c81414fB0D47
        _transferUsds(WBC, 3_613 * WAD);

        // PBG - 3,429 USDS - 0x8D4df847dB7FfE0B46AF084fE031F7691C6478c2
        _transferUsds(PBG, 3_429 * WAD);

        // Byteron - 571 USDS - 0xc2982e72D060cab2387Dba96b846acb8c96EfF66
        _transferUsds(BYTERON, 571 * WAD);

        // ---------- Atlas Core Development March 2025 USDS Payments ----------
        // Forum: https://forum.sky.money/t/atlas-core-development-payment-requests-march-2025/26077
        // Forum: https://forum.sky.money/t/atlas-core-development-payment-requests-february-2025/25921/6

        // BLUE - 50,167 USDS - 0xb6C09680D822F162449cdFB8248a7D3FC26Ec9Bf
        _transferUsds(BLUE, 50_167 * WAD);

        // Cloaky - 16,417 USDS - 0x9244F47D70587Fa2329B89B6f503022b63Ad54A5
        _transferUsds(CLOAKY_2, 16_417 * WAD);

        // Kohla - 10,000 USDS - 0x73dFC091Ad77c03F2809204fCF03C0b9dccf8c7a
        _transferUsds(CLOAKY_KOHLA_2, 10_000 * WAD);

        // Ennoia - 10,055 USDS - 0xA7364a1738D0bB7D1911318Ca3FB3779A8A58D7b
        _transferUsds(CLOAKY_ENNOIA, 10_055 * WAD);

        // ---------- Atlas Core Development March 2025 SKY Payments ----------
        // Forum: https://forum.sky.money/t/atlas-core-development-payment-requests-march-2025/26077
        // Forum: https://forum.sky.money/t/atlas-core-development-payment-requests-february-2025/25921/6

        // BLUE - 330,000 SKY - 0xb6C09680D822F162449cdFB8248a7D3FC26Ec9Bf
        _transferSky(BLUE, 330_000 * WAD);

        // Cloaky - 288,000 SKY - 0x9244F47D70587Fa2329B89B6f503022b63Ad54A5
        _transferSky(CLOAKY_2, 288_000 * WAD);

        // ---------- Top-up of the Integration Boost ----------
        // Forum: https://forum.sky.money/t/utilization-of-the-integration-boost-budget-a-5-2-1-2/25536/8
        // Atlas: https://sky-atlas.powerhouse.io/A.5.2.1.2_Integration_Boost/129f2ff0-8d73-8057-850b-d32304e9c91a%7C8d5a9e88cf49

        // Integration Boost - 3,000,000 USDS - 0xD6891d1DFFDA6B0B1aF3524018a1eE2E608785F7
        _transferUsds(INTEGRATION_BOOST_INITIATIVE, 3_000_000 * WAD);

        // ---------- Trigger Spark Proxy Spell ----------
        // Forum: https://forum.sky.money/t/mar-20-2025-stability-scope-parameter-changes-24/26129
        // Forum: https://forum.sky.money/t/mar-20-2025-stability-scope-parameter-changes-24/26129/2
        // Forum: https://forum.sky.money/t/march-6-2025-proposed-changes-to-spark-for-upcoming-spell/26036
        // Forum: https://forum.sky.money/t/march-20-2025-proposed-changes-to-spark-for-upcoming-spell/26113
        // Poll: https://vote.makerdao.com/polling/QmQrGdQz
        // Poll: https://vote.makerdao.com/polling/QmfM4SBB
        // Poll: https://vote.makerdao.com/polling/QmbDzZ3F
        // Poll: https://vote.makerdao.com/polling/Qmf4PDcJ#vote-breakdown
        // Poll: https://vote.makerdao.com/polling/QmXvuNAv
        // Poll: https://vote.makerdao.com/polling/QmXrHgdj
        // Poll: https://vote.makerdao.com/polling/QmTj3BSu
        // Poll: https://vote.makerdao.com/polling/QmPkA2GP

        // Execute Spark Spell at 0x1e865856d8F97FB34FBb0EDbF63f53E29a676aB6
        ProxyLike(SPARK_PROXY).exec(SPARK_SPELL, abi.encodeWithSignature("execute()"));
    }

    // ---------- Helper Functions ----------

    /// @notice wraps the operations required to transfer USDS from the surplus buffer.
    /// @param usr The USDS receiver.
    /// @param wad The USDS amount in wad precision (10 ** 18).
    function _transferUsds(address usr, uint256 wad) internal {
        // Note: Enforce whole units to avoid rounding errors
        require(wad % WAD == 0, "transferUsds/non-integer-wad");
        // Note: DssExecLib currently only supports Dai transfers from the surplus buffer.
        DssExecLib.sendPaymentFromSurplusBuffer(address(this), wad / WAD);
        // Note: Approve DAI_USDS for the amount sent to be able to convert it.
        DAI.approve(DAI_USDS, wad);
        // Note: Convert Dai to USDS for `usr`.
        DaiUsdsLike(DAI_USDS).daiToUsds(usr, wad);
    }

    /// @notice wraps the operations required to transfer SKY from the treasury.
    /// @param usr The SKY receiver.
    /// @param wad The SKY amount in wad precision (10 ** 18).
    function _transferSky(address usr, uint256 wad) internal {
        // Note: Calculate the equivalent amount of MKR required
        uint256 mkrWad = wad / MKR_SKY_RATE;
        // Note: if rounding error is expected, add an extra wei of MKR
        if (wad % MKR_SKY_RATE != 0) { mkrWad++; }
        // Note: Approve MKR_SKY for the amount sent to be able to convert it
        MKR.approve(MKR_SKY, mkrWad);
        // Note: Convert the calculated amount to SKY for `PAUSE_PROXY`
        MkrSkyLike(MKR_SKY).mkrToSky(address(this), mkrWad);
        // Note: Transfer originally requested amount, leaving extra on the `PAUSE_PROXY`
        GemAbstract(SKY).transfer(usr, wad);
    }
}

contract DssSpell is DssExec {
    constructor() DssExec(block.timestamp + 30 days, address(new DssSpellAction())) {}
}
