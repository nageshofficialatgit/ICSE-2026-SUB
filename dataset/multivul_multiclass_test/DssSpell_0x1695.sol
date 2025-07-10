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

// lib/dss-test/lib/dss-interfaces/src/dss/DaiJoinAbstract.sol

// https://github.com/makerdao/dss/blob/master/src/join.sol
interface DaiJoinAbstract {
    function wards(address) external view returns (uint256);
    function rely(address usr) external;
    function deny(address usr) external;
    function vat() external view returns (address);
    function dai() external view returns (address);
    function live() external view returns (uint256);
    function cage() external;
    function join(address, uint256) external;
    function exit(address, uint256) external;
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

interface DaiUsdsLike {
    function daiToUsds(address usr, uint256 wad) external;
}

interface SUsdsLike {
    function drip() external returns (uint256);
    function file(bytes32 what, uint256 data) external;
}

interface ProxyLike {
    function exec(address target, bytes calldata args) external payable returns (bytes memory out);
}

contract DssSpellAction is DssAction {
    // Provides a descriptive tag for bot consumption
    // This should be modified weekly to provide a summary of the actions
    // Hash: cast keccak -- "$(wget 'https://raw.githubusercontent.com/makerdao/community/718790846590c531038a3ebd58fbea9a48a0f293/governance/votes/Executive%20vote%20-%20February%206%2C%202025.md' -q -O - 2>/dev/null)"
    string public constant override description = "2025-02-06 MakerDAO Executive Spell | Hash: 0xc89ebc18cbe250c96a4a72bdbdb45cf258b6634597b119fdb1fd0969e79cf629";

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
    uint256 internal constant ONE_PT_THREE_THREE_PCT_RATE     = 1000000000418960282689704878;  //  1.33%
    uint256 internal constant SEVEN_PT_TWO_FIVE_PCT_RATE      = 1000000002219443553326580536;  //  7.25%
    uint256 internal constant EIGHT_PT_SEVEN_FIVE_PCT_RATE    = 1000000002659864411854984565;  //  8.75%
    uint256 internal constant NINE_PT_FIVE_PCT_RATE           = 1000000002877801985002875644;  //  9.50%
    uint256 internal constant NINE_PT_SEVEN_FIVE_PCT_RATE     = 1000000002950116251408586949;  //  9.75%
    uint256 internal constant TEN_PT_TWO_FIVE_PCT_RATE        = 1000000003094251918120023627;  // 10.25%
    uint256 internal constant TEN_PT_FIVE_PCT_RATE            = 1000000003166074807451009595;  // 10.50%
    uint256 internal constant TEN_PT_SEVEN_FIVE_PCT_RATE      = 1000000003237735385034516037;  // 10.75%
    uint256 internal constant FOURTEEN_PCT_RATE               = 1000000004154878953532704765;  // 14.00%
    uint256 internal constant FOURTEEN_PT_TWO_FIVE_PCT_RATE   = 1000000004224341833701283597;  // 14.25%
    uint256 internal constant FOURTEEN_PT_SEVEN_FIVE_PCT_RATE = 1000000004362812761691191350;  // 14.75%

    // ---------- Math ----------
    uint256 internal constant WAD = 10 ** 18;
    uint256 internal constant RAY = 10 ** 27;

    // ---------- Contracts ----------
    GemAbstract internal immutable DAI      = GemAbstract(DssExecLib.dai());
    GemAbstract internal immutable MKR      = GemAbstract(DssExecLib.mkr());
    address internal immutable DAI_USDS     = DssExecLib.getChangelogAddress("DAI_USDS");
    address internal immutable SUSDS        = DssExecLib.getChangelogAddress("SUSDS");
    address internal immutable MCD_JOIN_DAI = DssExecLib.daiJoin();
    address internal immutable MCD_VAT      = DssExecLib.vat();
    address internal immutable MCD_VOW      = DssExecLib.vow();

    // ---------- Wallets ----------
    address internal constant INTEGRATION_BOOST_INITIATIVE = 0xD6891d1DFFDA6B0B1aF3524018a1eE2E608785F7;
    address internal constant GFXLABS                      = 0xa6e8772af29b29B9202a073f8E36f447689BEef6;

    // ---------- Spark Proxy Spell ----------
    // Spark Proxy: https://github.com/marsfoundation/sparklend-deployments/blob/bba4c57d54deb6a14490b897c12a949aa035a99b/script/output/1/primary-sce-latest.json#L2
    address internal constant SPARK_PROXY = 0x3300f198988e4C9C63F75dF86De36421f06af8c4;
    address internal constant SPARK_SPELL = 0xD5c59b7c1DD8D2663b4c826574ed968B2C8329C0;

    function actions() public override {
        // ---------- Rate Adjustments ----------
        // Forum: https://forum.sky.money/t/feb-6-2025-stability-scope-parameter-changes-21/25906
        // Forum: https://forum.sky.money/t/feb-6-2025-stability-scope-parameter-changes-21/25906/3
        // Forum: https://forum.sky.money/t/feb-6-2025-stability-scope-parameter-changes-21/25906/4

        // Reduce ETH-A Stability Fee by 3 percentage points from 12.75% to 9.75%
        DssExecLib.setIlkStabilityFee("ETH-A", NINE_PT_SEVEN_FIVE_PCT_RATE, /* doDrip = */ true);

        // Reduce ETH-B Stability Fee by 3 percentage points from 13.25% to 10.25%
        DssExecLib.setIlkStabilityFee("ETH-B", TEN_PT_TWO_FIVE_PCT_RATE, /* doDrip = */ true);

        // Reduce ETH-C Stability Fee by 3 percentage points from 12.50% to 9.50%
        DssExecLib.setIlkStabilityFee("ETH-C", NINE_PT_FIVE_PCT_RATE, /* doDrip = */ true);

        // Reduce WSTETH-A Stability Fee by 3 percentage points from 13.75% to 10.75%
        DssExecLib.setIlkStabilityFee("WSTETH-A", TEN_PT_SEVEN_FIVE_PCT_RATE, /* doDrip = */ true);

        // Reduce WSTETH-B Stability Fee by 3 percentage points from 13.50% to 10.50%
        DssExecLib.setIlkStabilityFee("WSTETH-B", TEN_PT_FIVE_PCT_RATE, /* doDrip = */ true);

        // Reduce WBTC-A Stability Fee by 2 percentage points from 16.25% to 14.25%
        DssExecLib.setIlkStabilityFee("WBTC-A", FOURTEEN_PT_TWO_FIVE_PCT_RATE, /* doDrip = */ true);

        // Reduce WBTC-B Stability Fee by 2 percentage points from 16.75% to 14.75%
        DssExecLib.setIlkStabilityFee("WBTC-B", FOURTEEN_PT_SEVEN_FIVE_PCT_RATE, /* doDrip = */ true);

        // Reduce WBTC-C Stability Fee by 2 percentage points from 16.00% to 14.00%
        DssExecLib.setIlkStabilityFee("WBTC-C", FOURTEEN_PCT_RATE, /* doDrip = */ true);

        // Reduce ALLOCATOR-SPARK-A Stability Fee by 4.04 percentage points from 5.37% to 1.33%
        DssExecLib.setIlkStabilityFee("ALLOCATOR-SPARK-A", ONE_PT_THREE_THREE_PCT_RATE, /* doDrip = */ true);

        // Reduce DSR by 4 percentage points from 11.25% to 7.25%
        DssExecLib.setDSR(SEVEN_PT_TWO_FIVE_PCT_RATE, /* doDrip = */ true);

        // Reduce SSR by 3.75 percentage points from 12.50% to 8.75%
        SUsdsLike(SUSDS).drip();
        SUsdsLike(SUSDS).file("ssr", EIGHT_PT_SEVEN_FIVE_PCT_RATE);

        // ---------- Sweep Dai from PauseProxy to Surplus Buffer ----------
        // Forum: https://forum.sky.money/t/consolfreight-rwa-003-cf4-drop-default/21745/23
        // Forum: https://forum.sky.money/t/consolfreight-rwa-003-cf4-drop-default/21745/24

        // Sweep 406,451.52 Dai returned by ConsolFreight from the PauseProxy to the Surplus Buffer
        // Note: Approve the DaiJoin for the amount returned
        DAI.approve(MCD_JOIN_DAI, 406_451.52 ether);
        // Note: Join the DaiJoin for the amount returned using the Vow as destination
        DaiJoinAbstract(MCD_JOIN_DAI).join(MCD_VOW, 406_451.52 ether);

        // ---------- Integration Boost Funding ----------
        // Forum: https://forum.sky.money/t/utilization-of-the-integration-boost-budget-a-5-2-1-2/25536/5
        // Atlas: https://sky-atlas.powerhouse.io/A.5.2.1.2_Integration_Boost/129f2ff0-8d73-8057-850b-d32304e9c91a%7C8d5a9e88cf49

        // Integration Boost - 3,000,000 USDS - 0xD6891d1DFFDA6B0B1aF3524018a1eE2E608785F7
        _transferUsds(INTEGRATION_BOOST_INITIATIVE, 3_000_000 * WAD);

        // ---------- Whistleblower Payment ----------
        // Forum: https://forum.sky.money/t/whistleblower-bounty-and-ad-penalty-for-misalignment/25911
        // Atlas: https://sky-atlas.powerhouse.io/A.1.5.19_Whistleblower_Bounty/fb2de9a9-8154-46b8-9631-a5dda875921e%7C0db3af4e955e

        // GFX Labs - 1,000 USDS - 0xa6e8772af29b29B9202a073f8E36f447689BEef6
        _transferUsds(GFXLABS, 1_000 * WAD);

        // ---------- Spark Proxy Spell ----------
        // Forum: https://forum.sky.money/t/feb-6-2025-proposed-changes-to-spark-for-upcoming-spell-actual/25888
        // Poll: https://vote.makerdao.com/polling/QmUMkWLQ
        // Poll: https://vote.makerdao.com/polling/QmTfntSm
        // Poll: https://vote.makerdao.com/polling/QmWCe4JD
        // Poll: https://vote.makerdao.com/polling/QmbSANrr
        // Poll: https://vote.makerdao.com/polling/QmRKhzad

        // Execute Spark Spell at 0xD5c59b7c1DD8D2663b4c826574ed968B2C8329C0
        // Note: Make sure to not revert the Core spell if the Spark spell reverts
        try ProxyLike(SPARK_PROXY).exec(SPARK_SPELL, abi.encodeWithSignature("execute()")) {} catch {}
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
}

contract DssSpell is DssExec {
    constructor() DssExec(block.timestamp + 30 days, address(new DssSpellAction())) {}
}
